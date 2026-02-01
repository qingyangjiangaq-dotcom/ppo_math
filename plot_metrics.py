import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_series(steps, values, title, ylabel, out_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, values, linewidth=1.5)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/ppo_math/metrics.jsonl")
    parser.add_argument("--out_dir", default="outputs/ppo_math/plots")
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    if not rows:
        raise SystemExit(f"No metrics found at {args.metrics}")

    os.makedirs(args.out_dir, exist_ok=True)

    steps = [r.get("step", i) for i, r in enumerate(rows)]

    series = {
        "mean_reward": ("Train/Mean Reward", "reward"),
        "std_reward": ("Train/Std Reward", "reward"),
        "parse_rate": ("Train/Parse Rate", "rate"),
        "format_rate": ("Train/Format Rate", "rate"),
        "approx_kl": ("Train/Approx KL", "kl"),
        "clip_frac": ("Train/Clip Fraction", "ratio"),
        "entropy": ("Train/Entropy", "entropy"),
        "policy_loss": ("Train/Policy Loss", "loss"),
        "value_loss": ("Train/Value Loss", "loss"),
        "total_loss": ("Train/Total Loss", "loss"),
        "explained_variance": ("Train/Explained Variance", "var"),
        "value_mean": ("Train/Value Mean", "value"),
        "value_std": ("Train/Value Std", "value"),
        "return_mean": ("Train/Return Mean", "return"),
        "raw_score_mean": ("Train/Raw Score Mean", "score"),
        "raw_score_std": ("Train/Raw Score Std", "score"),
        "advantage_mean": ("Train/Advantage Mean", "adv"),
        "response_len": ("Train/Response Length", "tokens"),
        "query_len": ("Train/Query Length", "tokens"),
        "ref_ppl": ("Text/Ref PPL", "ppl"),
        "distinct2": ("Text/Distinct-2", "ratio"),
        "distinct3": ("Text/Distinct-3", "ratio"),
        "step_time_sec": ("Perf/Step Time", "sec"),
        "gen_tokens_per_sec": ("Perf/Tokens Per Sec", "tok/s"),
        "samples_per_sec": ("Perf/Samples Per Sec", "samples/s"),
        "learning_rate": ("Train/Learning Rate", "lr"),
        "gpu_mem_gb": ("Perf/GPU Mem", "GB"),
    }

    for key, (title, ylabel) in series.items():
        values = [r.get(key) for r in rows if r.get(key) is not None]
        if not values:
            continue
        plot_steps = [s for s, r in zip(steps, rows) if r.get(key) is not None]
        out_path = os.path.join(args.out_dir, f"{key}.png")
        plot_series(plot_steps, values, title, ylabel, out_path)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
