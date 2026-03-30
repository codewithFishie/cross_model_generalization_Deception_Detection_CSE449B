from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-domain-out cross-generalization runs")
    parser.add_argument("--manifest", default="data/manifests/all_data.csv")
    parser.add_argument("--out_dir", default="outputs/cross_domain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    domains = sorted(df["domain"].unique().tolist())

    results = []
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for holdout in domains:
        fold_dir = out_root / f"holdout_{holdout}"
        split_dir = fold_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        test_df = df[df["domain"] == holdout]
        trainval_df = df[df["domain"] != holdout].sample(frac=1.0, random_state=42)

        n_val = max(1, int(0.15 * len(trainval_df)))
        val_df = trainval_df.iloc[:n_val]
        train_df = trainval_df.iloc[n_val:]

        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        exp_dir = fold_dir / "run"
        run(
            [
                "python",
                "-m",
                "deception.train",
                "--train_csv",
                str(train_csv),
                "--val_csv",
                str(val_csv),
                "--test_csv",
                str(test_csv),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_domains",
                str(max(1, len(domains) - 1)),
                "--out_dir",
                str(exp_dir),
            ]
        )

        metrics = json.loads((exp_dir / "metrics.json").read_text())
        results.append(
            {
                "holdout": holdout,
                "test_acc": metrics["test_acc"],
                "test_f1": metrics["test_f1"],
                "best_val_f1": metrics["best_val_f1"],
            }
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_root / "summary.csv", index=False)
    print(res_df)
    print("Mean accuracy:", res_df["test_acc"].mean())


if __name__ == "__main__":
    main()
