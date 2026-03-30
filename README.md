# Cross-Model Generalization for Deception Detection

This repository now contains an end-to-end pipeline for **cross-domain deception detection** with explicit domain generalization training using a **Domain-Adversarial Neural Network (DANN)** setup.

## What this solves

Instead of only optimizing in-domain accuracy, the pipeline trains one encoder to:
1. predict deception labels (`truth` vs `lie`), and
2. remove dataset/domain-specific bias via an adversarial domain head.

This usually improves robustness when evaluated on a different dataset/theme.

---

## Expected folder structure

Your screenshots show this style:

```text
DOLOS_Train/
  lie/
  truth/

RLT_Test/
  lie/
  truth/
```

You can also include a third dataset with the same layout.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 1) Build merged manifest

```bash
python scripts/prepare_data.py \
  --dolos_root /path/to/DOLOS_Train \
  --rlt_root /path/to/RLT_Test \
  --third_root /optional/path/to/THIRD_DATASET \
  --output data/manifests/all_data.csv
```

Output columns:
- `path` (absolute image path)
- `label` (`truth=0`, `lie=1`)
- `label_name`
- `domain` (`dolos`, `rlt`, or third dataset name)

---

## 2) Run cross-domain experiment (leave-one-domain-out)

```bash
python scripts/run_cross_domain.py \
  --manifest data/manifests/all_data.csv \
  --out_dir outputs/cross_domain \
  --epochs 20 \
  --batch_size 16
```

For each domain:
- hold out that domain as test set,
- train on the others,
- validate on a subset of train domains,
- report held-out test metrics.

Final summary is saved to:

```text
outputs/cross_domain/summary.csv
```

---

## 3) Single custom run

```bash
python -m deception.train \
  --train_csv outputs/cross_domain/holdout_rlt/splits/train.csv \
  --val_csv outputs/cross_domain/holdout_rlt/splits/val.csv \
  --test_csv outputs/cross_domain/holdout_rlt/splits/test.csv \
  --epochs 30 \
  --batch_size 16 \
  --domain_loss_weight 0.2 \
  --num_domains 2 \
  --out_dir outputs/my_run
```

Metrics are written to:

```text
outputs/my_run/metrics.json
```

---

## 4) Evaluate a trained checkpoint

```bash
python -m deception.evaluate \
  --checkpoint outputs/my_run/best_model.pt \
  --test_csv outputs/cross_domain/holdout_rlt/splits/test.csv \
  --num_domains 2
```

---

## Notes on the 80% target

- This code is designed to maximize cross-domain generalization, but no static codebase can guarantee 80% on all unseen domains without seeing the real data distribution.
- If current accuracy is below 80%, first tune:
  - `epochs` (20 -> 40),
  - `image_size` (224 -> 320 if GPU allows),
  - `domain_loss_weight` (0.1 to 0.5),
  - stronger augmentations and class balancing.

A practical next step is hyperparameter sweeps over the above on leave-one-domain-out folds.

---

## Files added

- `src/deception/data.py` – data loading, transforms, manifests.
- `src/deception/model.py` – DANN model with gradient reversal.
- `src/deception/train.py` – training loop + checkpointing + metrics.
- `src/deception/evaluate.py` – standalone evaluation script.
- `scripts/prepare_data.py` – manifest generation from dataset folders.
- `scripts/run_cross_domain.py` – leave-one-domain-out automation.
