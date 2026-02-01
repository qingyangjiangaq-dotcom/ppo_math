import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

from utils import extract_last_number, normalize_answer, extract_gold_answer
from reward import parse_pred, compute_reward


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "swulling/gsm8k_chinese"
    dataset_config: str = "default"
    train_size: int = 2000
    eval_size: int = 200
    max_prompt_length: int = 256
    max_new_tokens: int = 512
    ppo_steps: int = 1000
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 16  # 4 * 16 = 64 有效 batch size，更稳定的梯度
    ppo_epochs: int = 2
    learning_rate: float = 1e-6  # 降低学习率防止参数乱跳
    init_kl_coef: float = 0.2  # 保持 KL 惩罚防止语言崩坏
    target_kl: float = 6.0
    seed: int = 42
    log_dir: str = "runs/ppo_math"
    output_dir: str = "outputs/ppo_math"
    eval_every: int = 100
    save_every: int = 200
    log_every: int = 10
    log_samples_every: int = 10


SYSTEM_PROMPT = (
    "你是一个数学解题助手。请分步推理（可用“首先/然后/所以/计算”等词），"
    "最后一行严格输出：'#### <数字>'，不要附加其他内容。"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(question: str, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _normalize_row(row):
    question = row.get("question_zh-cn") or row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("answer_only") or row.get("output")
    return {"question": question, "answer": answer}


def load_data(cfg: TrainConfig):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    test_split = "test" if "test" in ds else train_split

    train_ds = ds[train_split].shuffle(seed=cfg.seed).map(
        _normalize_row, remove_columns=ds[train_split].column_names
    )
    test_ds = ds[test_split].shuffle(seed=cfg.seed).map(
        _normalize_row, remove_columns=ds[test_split].column_names
    )

    train_ds = train_ds.select(range(min(cfg.train_size, len(train_ds))))
    test_ds = test_ds.select(range(min(cfg.eval_size, len(test_ds))))

    return train_ds, test_ds




def extract_pred_gold(response: str, answer_text: str):
    pred_raw, parse_ok, format_ok = parse_pred(response)
    pred = normalize_answer(pred_raw)
    gold = normalize_answer(extract_gold_answer(answer_text))
    return pred, gold, parse_ok, format_ok


def strip_prompt(response: str, prompt: str):
    if response.startswith(prompt):
        return response[len(prompt):]
    return response


def decode_responses(tokenizer, query_tensors, response_tensors):
    responses = []
    for query_ids, resp_ids in zip(query_tensors, response_tensors):
        q_len = query_ids.shape[-1]
        if resp_ids.shape[-1] >= q_len and torch.equal(resp_ids[:q_len], query_ids):
            resp_ids = resp_ids[q_len:]
        responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
    return responses


def _get_stat(stats, *keys, default=None):
    for key in keys:
        if key in stats:
            return stats[key]
    return default


def _to_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_distinct_n(texts, n=2):
    total = 0
    unique = set()
    for text in texts:
        tokens = text.strip().split()
        if len(tokens) < n:
            continue
        total += max(len(tokens) - n + 1, 0)
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i : i + n]))
    if total == 0:
        return 0.0
    return len(unique) / total


def compute_ref_ppl(ref_model, tokenizer, prompts, responses, device, max_length):
    ref_model.eval()
    losses = []
    token_counts = []
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            text = prompt + response
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).input_ids[0]
            labels = enc.input_ids.clone()
            labels[0, : prompt_ids.shape[0]] = -100
            outputs = ref_model(**enc, labels=labels)
            loss = outputs.loss
            valid_tokens = (labels != -100).sum().item()
            losses.append(loss.item())
            token_counts.append(valid_tokens)
    if not losses or not sum(token_counts):
        return None
    avg_loss = float(np.average(losses, weights=token_counts))
    return float(np.exp(avg_loss))


def _fmt(value, fmt="{:.3f}", default="n/a"):
    if value is None:
        return default
    return fmt.format(value)


def _prepare_scores(ppo_trainer, scores):
    scores = torch.tensor(scores, device=ppo_trainer.current_device)
    if ppo_trainer.config.use_score_scaling:
        running = ppo_trainer.running
        saved = (running.mean, running.std, running.var, running.count)
        scores_mean, scores_std = running.update(scores)
        tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        score_scaling_factor = running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        if ppo_trainer.config.use_score_norm:
            scores = (scores - running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        else:
            scores /= score_scaling_factor
        running.mean, running.std, running.var, running.count = saved
    if ppo_trainer.config.score_clip is not None:
        scores_dtype = scores.dtype
        scores = torch.clip(scores.float(), -ppo_trainer.config.score_clip, ppo_trainer.config.score_clip).to(
            dtype=scores_dtype
        )
    return scores


def compute_extra_stats(ppo_trainer, queries, responses, scores):
    # Pre-step stats
    model_inputs = ppo_trainer.prepare_model_inputs(queries, responses)
    with torch.no_grad():
        old_logprobs, _, old_values, masks = ppo_trainer.batched_forward_pass(
            ppo_trainer.model,
            queries,
            responses,
            model_inputs,
            response_masks=None,
            return_logits=False,
        )
        ref_logprobs, _, _, _ = ppo_trainer.batched_forward_pass(
            ppo_trainer.model if ppo_trainer.is_peft_model else ppo_trainer.ref_model,
            queries,
            responses,
            model_inputs,
            response_masks=None,
            return_logits=False,
        )
        rewards, _, _ = ppo_trainer.compute_rewards(_prepare_scores(ppo_trainer, scores), old_logprobs, ref_logprobs, masks)
        values, advantages, returns = ppo_trainer.compute_advantages(old_values, rewards, masks)

    # Post-step stats
    with torch.no_grad():
        logprobs, logits, vpreds, _ = ppo_trainer.batched_forward_pass(
            ppo_trainer.model,
            queries,
            responses,
            model_inputs,
            response_masks=None,
            return_logits=True,
        )

    _, _, flat_stats = ppo_trainer.loss(
        old_logprobs=old_logprobs,
        values=old_values,
        logits=logits,
        vpreds=vpreds,
        logprobs=logprobs,
        mask=masks,
        advantages=advantages,
        returns=returns,
    )
    return flat_stats


def evaluate(model, tokenizer, eval_ds, cfg: TrainConfig, device):
    model.eval()
    correct = 0
    total = 0
    for row in eval_ds:
        prompt = build_prompt(row["question"], tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
            padding=False,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        reward, _, _, _ = compute_reward(response, row["answer"])
        correct += 1.0 if reward > 0.5 else 0.0
        total += 1
    model.train()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--ppo_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mini_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--log_samples_every", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=None)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.train_size is not None:
        cfg.train_size = args.train_size
    if args.eval_size is not None:
        cfg.eval_size = args.eval_size
    if args.max_prompt_length is not None:
        cfg.max_prompt_length = args.max_prompt_length
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.ppo_steps is not None:
        cfg.ppo_steps = args.ppo_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.mini_batch_size is not None:
        cfg.mini_batch_size = args.mini_batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.log_samples_every is not None:
        cfg.log_samples_every = args.log_samples_every
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.init_kl_coef is not None:
        cfg.init_kl_coef = args.init_kl_coef
    if args.target_kl is not None:
        cfg.target_kl = args.target_kl
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    train_ds, eval_ds = load_data(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成任务通常左填充

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.model_name,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    device = next(model.parameters()).device

    ref_model = create_reference_model(model)

    ppo_config = PPOConfig(
        model_name=cfg.model_name,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        ppo_epochs=cfg.ppo_epochs,
        init_kl_coef=cfg.init_kl_coef,
        target_kl=cfg.target_kl,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    generation_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    start = time.time()
    step = 0
    data_iter = iter(train_ds)
    stats_keys_logged = False
    need_extra_stats = False

    while step < cfg.ppo_steps:
        step_start = time.time()
        batch = []
        for _ in range(cfg.batch_size):
            try:
                row = next(data_iter)
            except StopIteration:
                data_iter = iter(train_ds.shuffle(seed=cfg.seed + step))
                row = next(data_iter)
            batch.append(row)

        prompts = [build_prompt(row["question"], tokenizer) for row in batch]
        query_tensors = [
            tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_prompt_length,
                padding=False,
            ).input_ids[0].to(device)
            for prompt in prompts
        ]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        responses = decode_responses(tokenizer, query_tensors, response_tensors)

        rewards = []
        preds = []
        golds = []
        parse_hits = 0
        format_hits = 0
        for prompt, response, row in zip(prompts, responses, batch):
            # Remove prompt prefix if present
            reward, pred_raw, parse_ok, format_ok = compute_reward(response, row["answer"])
            pred, gold, _, _ = extract_pred_gold(response, row["answer"])
            if parse_ok:
                parse_hits += 1
            if format_ok:
                format_hits += 1
            rewards.append(torch.tensor(reward, device=device))
            preds.append(pred)
            golds.append(gold)

        extra_stats = None
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % cfg.log_every == 0:
            if not stats_keys_logged:
                keys_path = os.path.join(cfg.output_dir, "stats_keys.json")
                with open(keys_path, "w") as f:
                    json.dump(sorted(list(stats.keys())), f, indent=2)
                stats_keys_logged = True

            rewards_np = np.array([r.item() for r in rewards], dtype=np.float32)
            mean_reward = float(rewards_np.mean()) if len(rewards_np) else 0.0
            std_reward = float(rewards_np.std()) if len(rewards_np) else 0.0
            parse_rate = parse_hits / max(len(rewards_np), 1)
            format_rate = format_hits / max(len(rewards_np), 1)

            approx_kl = _to_float(
                _get_stat(
                    stats,
                    "ppo/policy/approxkl",
                    "policy/approxkl",
                    "policy/approx_kl",
                    "approx_kl",
                    "objective/kl",
                )
            )
            entropy = _to_float(
                _get_stat(stats, "ppo/policy/entropy", "policy/entropy", "objective/entropy")
            )
            clip_frac = _to_float(_get_stat(stats, "ppo/policy/clipfrac", "policy/clipfrac", "clipfrac"))
            policy_loss = _to_float(_get_stat(stats, "ppo/loss/policy", "loss/policy", "policy_loss"))
            value_loss = _to_float(_get_stat(stats, "ppo/loss/value", "loss/value", "value_loss", "vf_loss"))
            total_loss = _to_float(_get_stat(stats, "ppo/loss/total", "loss/total", "total_loss"))
            explained_var = _to_float(
                _get_stat(
                    stats,
                    "ppo/val/var_explained",
                    "val/var_explained",
                    "explained_variance",
                )
            )
            value_mean = _to_float(_get_stat(stats, "ppo/val/mean", "val/mean", "value_mean"))
            value_var = _to_float(_get_stat(stats, "ppo/val/var", "val/var", "value_var"))
            value_std = float(np.sqrt(value_var)) if value_var is not None else None
            return_mean = _to_float(_get_stat(stats, "ppo/returns/mean", "returns/mean", "return_mean"))
            raw_score_mean = _to_float(_get_stat(stats, "ppo/mean_scores", "mean_scores"))
            raw_score_std = _to_float(_get_stat(stats, "ppo/std_scores", "std_scores"))
            advantage_mean = _to_float(
                _get_stat(stats, "ppo/policy/advantages_mean", "policy/advantages_mean", "advantage_mean")
            )

            missing_core = any(
                v is None
                for v in [
                    approx_kl,
                    clip_frac,
                    entropy,
                    policy_loss,
                    value_loss,
                    total_loss,
                    explained_var,
                    return_mean,
                    advantage_mean,
                ]
            )
            if missing_core:
                need_extra_stats = True
                extra_stats = compute_extra_stats(ppo_trainer, query_tensors, response_tensors, rewards)
                if extra_stats:
                    stats = {**stats, **extra_stats}
                    approx_kl = _to_float(_get_stat(stats, "policy/approxkl", "ppo/policy/approxkl"))
                    entropy = _to_float(_get_stat(stats, "policy/entropy", "ppo/policy/entropy"))
                    clip_frac = _to_float(_get_stat(stats, "policy/clipfrac", "ppo/policy/clipfrac"))
                    policy_loss = _to_float(_get_stat(stats, "loss/policy", "ppo/loss/policy"))
                    value_loss = _to_float(_get_stat(stats, "loss/value", "ppo/loss/value"))
                    total_loss = _to_float(_get_stat(stats, "loss/total", "ppo/loss/total"))
                    explained_var = _to_float(
                        _get_stat(stats, "val/var_explained", "ppo/val/var_explained")
                    )
                    value_mean = _to_float(_get_stat(stats, "val/mean", "ppo/val/mean"))
                    value_var = _to_float(_get_stat(stats, "val/var", "ppo/val/var"))
                    value_std = float(np.sqrt(value_var)) if value_var is not None else None
                    return_mean = _to_float(_get_stat(stats, "returns/mean", "ppo/returns/mean"))
                    advantage_mean = _to_float(_get_stat(stats, "policy/advantages_mean", "ppo/policy/advantages_mean"))

            writer.add_scalar("train/mean_reward", mean_reward, step)
            writer.add_scalar("train/std_reward", std_reward, step)
            writer.add_scalar("train/parse_rate", parse_rate, step)
            writer.add_scalar("train/format_rate", format_rate, step)
            if approx_kl is not None:
                writer.add_scalar("train/approx_kl", approx_kl, step)
            if entropy is not None:
                writer.add_scalar("train/entropy", entropy, step)
            if clip_frac is not None:
                writer.add_scalar("train/clip_frac", clip_frac, step)
            if policy_loss is not None:
                writer.add_scalar("train/policy_loss", policy_loss, step)
            if value_loss is not None:
                writer.add_scalar("train/value_loss", value_loss, step)
            if total_loss is not None:
                writer.add_scalar("train/total_loss", total_loss, step)
            if explained_var is not None:
                writer.add_scalar("train/explained_variance", explained_var, step)
            if value_mean is not None:
                writer.add_scalar("train/value_mean", value_mean, step)
            if value_std is not None:
                writer.add_scalar("train/value_std", value_std, step)
            if return_mean is not None:
                writer.add_scalar("train/return_mean", return_mean, step)
            if raw_score_mean is not None:
                writer.add_scalar("train/raw_score_mean", raw_score_mean, step)
            if raw_score_std is not None:
                writer.add_scalar("train/raw_score_std", raw_score_std, step)
            if advantage_mean is not None:
                writer.add_scalar("train/advantage_mean", advantage_mean, step)

            with torch.no_grad():
                pad_id = tokenizer.pad_token_id
                resp_lens = np.array(
                    [int((t != pad_id).sum().item()) for t in response_tensors], dtype=np.float32
                )
                query_lens = np.array(
                    [int((t != pad_id).sum().item()) for t in query_tensors], dtype=np.float32
                )
            writer.add_scalar("train/response_len", float(resp_lens.mean()), step)
            writer.add_scalar("train/query_len", float(query_lens.mean()), step)

            step_time = time.time() - step_start
            writer.add_scalar("perf/step_time_sec", step_time, step)
            gen_tokens = float(resp_lens.sum())
            writer.add_scalar("perf/gen_tokens_per_sec", gen_tokens / max(step_time, 1e-6), step)
            samples_per_sec = cfg.batch_size / max(step_time, 1e-6)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, step)
            lr = None
            if hasattr(ppo_trainer, "optimizer") and ppo_trainer.optimizer is not None:
                lr = ppo_trainer.optimizer.param_groups[0].get("lr")
                if lr is not None:
                    writer.add_scalar("train/learning_rate", lr, step)
            gpu_mem_gb = None
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                writer.add_scalar("perf/gpu_mem_gb", gpu_mem_gb, step)

            ref_ppl = None
            try:
                ref_ppl = compute_ref_ppl(
                    ref_model.pretrained_model if hasattr(ref_model, "pretrained_model") else ref_model,
                    tokenizer,
                    prompts,
                    responses,
                    device,
                    max_length=min(cfg.max_prompt_length + cfg.max_new_tokens, 512),
                )
            except Exception:
                ref_ppl = None

            distinct2 = compute_distinct_n(responses, n=2)
            distinct3 = compute_distinct_n(responses, n=3)

            if ref_ppl is not None:
                writer.add_scalar("text/ref_ppl", ref_ppl, step)
            writer.add_scalar("text/distinct2", distinct2, step)
            writer.add_scalar("text/distinct3", distinct3, step)

            metrics = {
                "step": step,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "parse_rate": parse_rate,
                "format_rate": format_rate,
                "approx_kl": approx_kl,
                "entropy": entropy,
                "clip_frac": clip_frac,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
                "explained_variance": explained_var,
                "value_mean": value_mean,
                "value_std": value_std,
                "return_mean": return_mean,
                "raw_score_mean": raw_score_mean,
                "raw_score_std": raw_score_std,
                "advantage_mean": advantage_mean,
                "response_len": float(resp_lens.mean()),
                "query_len": float(query_lens.mean()),
                "step_time_sec": float(step_time),
                "gen_tokens_per_sec": float(gen_tokens / max(step_time, 1e-6)),
                "samples_per_sec": float(samples_per_sec),
                "learning_rate": lr,
                "ref_ppl": ref_ppl,
                "distinct2": distinct2,
                "distinct3": distinct3,
            }
            if gpu_mem_gb is not None:
                metrics["gpu_mem_gb"] = float(gpu_mem_gb)

            metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            print(
                " | ".join(
                    [
                        f"step {step}",
                        f"reward {mean_reward:.3f}±{std_reward:.3f}",
                        f"parse {parse_rate:.2%}",
                        f"fmt {format_rate:.2%}",
                        f"kl {_fmt(metrics['approx_kl'])}",
                        f"clip {_fmt(metrics['clip_frac'])}",
                        f"ent {_fmt(metrics['entropy'])}",
                        f"len q{metrics['query_len']:.1f}/r{metrics['response_len']:.1f}",
                        f"t {metrics['step_time_sec']:.2f}s",
                        f"tok/s {metrics['gen_tokens_per_sec']:.1f}",
                        f"ppl {_fmt(metrics['ref_ppl'], '{:.2f}')}",
                        f"d2 {_fmt(metrics['distinct2'])}",
                    ]
                    + ([f"mem {metrics['gpu_mem_gb']:.2f}GB"] if "gpu_mem_gb" in metrics else [])
                )
            )

        if step % cfg.eval_every == 0 and step > 0:
            acc = evaluate(model.pretrained_model, tokenizer, eval_ds, cfg, device)
            writer.add_scalar("eval/accuracy", acc, step)

        if step % cfg.log_samples_every == 0:
            sample_path = os.path.join(cfg.output_dir, "samples.jsonl")
            with open(sample_path, "a") as f:
                for prompt, response, reward, pred, gold, row in zip(
                    prompts, responses, rewards, preds, golds, batch
                ):
                    rec = {
                        "step": step,
                        "question": row["question"],
                        "answer": row["answer"],
                        "response": response,
                        "pred": pred,
                        "gold": gold,
                        "reward": float(reward.item()),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for prompt, response, pred, gold, row in zip(prompts, responses, preds, golds, batch):
                print("sample | question:", row["question"])
                print("sample | response:", response)
                print("sample | pred:", pred, "| gold:", gold)

        if step % cfg.save_every == 0 and step > 0:
            save_path = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(save_path, exist_ok=True)
            ppo_trainer.save_pretrained(save_path)

        step += 1

    total_time = time.time() - start
    final_acc = evaluate(model.pretrained_model, tokenizer, eval_ds, cfg, device)
    with open(os.path.join(cfg.output_dir, "final_metrics.json"), "w") as f:
        json.dump({"final_accuracy": final_acc, "total_time_sec": total_time}, f, indent=2)

    print(f"Final accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
