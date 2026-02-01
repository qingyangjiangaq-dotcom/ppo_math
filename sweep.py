import itertools
import subprocess
import time


def main():
    # Small sweep to finish within a few hours on 8GB VRAM
    learning_rates = [5e-6, 1e-5]
    kl_coefs = [0.1, 0.2]
    steps = [600, 1000]

    combos = list(itertools.product(learning_rates, kl_coefs, steps))

    for lr, kl, ppo_steps in combos:
        run_name = f"lr{lr}_kl{kl}_steps{ppo_steps}"
        cmd = [
            "python",
            "train_ppo.py",
            "--learning_rate",
            str(lr),
            "--init_kl_coef",
            str(kl),
            "--ppo_steps",
            str(ppo_steps),
        ]
        print(f"\n=== Running {run_name} ===")
        start = time.time()
        subprocess.run(cmd, check=True)
        print(f"=== Finished {run_name} in {time.time() - start:.1f}s ===")


if __name__ == "__main__":
    main()
