from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from deception.data import DatasetConfig, make_dataloaders
from deception.model import DeceptionDANN, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_domains", type=int, default=2)
    args = parser.parse_args()

    cfg = DatasetConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_csv=args.test_csv,
        val_csv=args.test_csv,
        test_csv=args.test_csv,
    )
    loader = make_dataloaders(cfg, domain_adversarial=True)["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeceptionDANN(ModelConfig(num_domains=args.num_domains)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits, _ = model(x, lambd=0.0)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
