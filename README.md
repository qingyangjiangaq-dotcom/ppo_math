# PPO 数学实验（GSM8K，几小时内）

适配环境：RTX 4060 8GB VRAM + 8GB RAM（WSL2）。

## 目标
- 使用 PPO 让小型 LLM 在 GSM8K 上提升答案正确率
- 奖励函数使用“格式 + 过程 + 结果”的混合奖励（稠密）
- 支持训练监控与小规模参数扫参

## 快速开始

```bash
cd /home/uincy/projects/mywsl/ppo_math
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

训练（默认约 1-3 小时，取决于步骤数）：

```bash
python train_ppo.py
```

评估（对保存的模型）：

```bash
python eval.py --model_path outputs/ppo_math/checkpoint-200
```

TensorBoard 监控：

```bash
tensorboard --logdir runs/ppo_math --port 6006
```

监控项（更详细）：
- train/mean_reward、train/std_reward、train/parse_rate、train/format_rate
- train/approx_kl、train/clip_frac、train/entropy
- train/policy_loss、train/value_loss、train/total_loss、train/explained_variance
- train/return_mean、train/raw_score_mean、train/raw_score_std、train/advantage_mean
- train/response_len、train/query_len、text/ref_ppl、text/distinct2、text/distinct3
- perf/step_time_sec、perf/gen_tokens_per_sec、perf/samples_per_sec、perf/gpu_mem_gb
- eval/accuracy

样本输出（jsonl）：`outputs/ppo_math/samples.jsonl`

小型扫参（可选）：

```bash
python sweep.py
```

## 训练说明
- 使用 `swulling/gsm8k_chinese` 数据集（中文数学）
- 模型默认：`Qwen/Qwen2.5-0.5B-Instruct`
- Prompt 要求：分步推理，最后一行严格输出 `#### <数字>`
- 奖励函数：格式奖励 + 过程奖励 + 结果奖励（与 prompt 严格一致）
- 评估指标：accuracy（精确匹配）

## 可调参数
常用参数（训练时覆盖）：

```bash
python train_ppo.py \
  --ppo_steps 800 \
  --batch_size 8 \
  --mini_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_new_tokens 512 \
  --learning_rate 1e-5 \
  --init_kl_coef 0.2
```

## 产物
- 模型：`outputs/ppo_math/checkpoint-*`
- 训练日志：`runs/ppo_math`
- 最终指标：`outputs/ppo_math/final_metrics.json`

## 注意
- 8GB 显存较紧，如果显存不足可降低 `batch_size` 或 `max_new_tokens`
- 如果运行变慢，可将 `train_size` 降到 1000
