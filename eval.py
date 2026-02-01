import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from reward import parse_pred, compute_reward


SYSTEM_PROMPT = (
    "你是一个数学解题助手。请分步推理（可用“首先/然后/所以/计算”等词），"
    "最后一行严格输出：'#### <数字>'，不要附加其他内容。"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--dataset_name", type=str, default="swulling/gsm8k_chinese")
    parser.add_argument("--dataset_config", type=str, default="default")
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name, args.dataset_config)
    split = "test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0])
    eval_ds = (
        ds[split]
        .shuffle(seed=42)
        .map(
            lambda row: {
                "question": row.get("question_zh-cn") or row.get("question") or row.get("prompt"),
                "answer": row.get("answer") or row.get("answer_only") or row.get("output"),
            },
            remove_columns=ds[split].column_names,
        )
        .select(range(min(args.eval_size, len(ds[split]))))
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成任务通常左填充

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    correct = 0
    total = 0
    for row in eval_ds:
        prompt = build_prompt(row["question"], tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        reward, _, _, _ = compute_reward(response, row["answer"])
        correct += 1.0 if reward > 0.5 else 0.0
        total += 1

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
