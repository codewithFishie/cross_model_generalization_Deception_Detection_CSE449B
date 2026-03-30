from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deception.data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create cross-domain manifests from folder datasets."
    )
    parser.add_argument("--dolos_root", required=True, help="Path to DOLOS folder containing lie/truth")
    parser.add_argument("--rlt_root", required=True, help="Path to RLT folder containing lie/truth")
    parser.add_argument("--third_root", default=None, help="Optional third dataset root")
    parser.add_argument("--output", default="data/manifests/all_data.csv")
    args = parser.parse_args()

    tmp_dir = Path("data/manifests")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    domain_manifests = []

    dolos_csv = tmp_dir / "dolos.csv"
    build_manifest(args.dolos_root, str(dolos_csv), domain_name="dolos")
    domain_manifests.append(pd.read_csv(dolos_csv))

    rlt_csv = tmp_dir / "rlt.csv"
    build_manifest(args.rlt_root, str(rlt_csv), domain_name="rlt")
    domain_manifests.append(pd.read_csv(rlt_csv))

    if args.third_root:
        third_name = Path(args.third_root).name.lower().replace(" ", "_")
        third_csv = tmp_dir / f"{third_name}.csv"
        build_manifest(args.third_root, str(third_csv), domain_name=third_name)
        domain_manifests.append(pd.read_csv(third_csv))

    merged = pd.concat(domain_manifests, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved merged manifest: {args.output} ({len(merged)} samples)")


if __name__ == "__main__":
    main()
