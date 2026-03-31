from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from deception.data import DatasetConfig, make_dataloaders
from deception.model import DeceptionDANN, ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    out_dir: str = "outputs/exp"
    device: str = "cuda"


def compute_lambda(epoch: int, total_epochs: int) -> float:
    progress = epoch / max(total_epochs - 1, 1)
    return 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0


def evaluate(model, loader, device, with_domain):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            if with_domain:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x, lambd=0.0)
            pred = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_domains", type=int, default=2)
    parser.add_argument("--domain_loss_weight", type=float, default=0.2)
    parser.add_argument("--out_dir", default="outputs/exp")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DatasetConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )
    model_cfg = ModelConfig(
        pretrained=True,
        num_domains=args.num_domains,
        domain_loss_weight=args.domain_loss_weight,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dir=args.out_dir,
        device=device,
    )

    loaders = make_dataloaders(data_cfg, domain_adversarial=True)

    model = DeceptionDANN(model_cfg).to(device)
    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)

    best_f1 = -1.0
    history = []
    for epoch in range(train_cfg.epochs):
        model.train()
        lambd = compute_lambda(epoch, train_cfg.epochs)
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{train_cfg.epochs}")
        total_loss = 0.0
        for x, y, d in pbar:
            x = x.to(device)
            y = y.to(device)
            d = d.to(device)

            cls_logits, dom_logits = model(x, lambd=lambd)
            cls_loss = cls_criterion(cls_logits, y)
            dom_loss = dom_criterion(dom_logits, d)
            loss = cls_loss + model_cfg.domain_loss_weight * dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / len(loaders["train"].dataset)
        val_metrics = evaluate(model, loaders["val"], device, with_domain=True)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, loaders["test"], device, with_domain=True)

    metrics = {
        "best_val_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],
        "history": history,
        "config": {
            "data": asdict(data_cfg),
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
