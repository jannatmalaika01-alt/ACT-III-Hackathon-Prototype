"""Train the defect classifier.   python -m vision.train [--per-class 700] [--epochs 12]

Data = generated images (vision/synthetic.py) + any real photos found in data/train/<class_id>/.
Validation always uses generated images from seeds the training set never sees. Writes
vision/weights/defect_cnn.pt and vision/weights/defect_cnn.json (classes + validation accuracy).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import synthetic
from .labels import CLASSES
from .model import DefectCNN
from .preprocess import to_array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "defect_cnn.pt")
META = WEIGHTS.replace(".pt", ".json")
TRAIN_SEED, VAL_SEED = 1_000, 10_000_000  # test_11..20 use 20_000_000+, see setup_dataset.py


def _real_images():
    xs, ys = [], []
    base = os.path.join(ROOT, "data", "train")
    for ci, c in enumerate(CLASSES):
        folder = os.path.join(base, c)
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    xs.append(to_array(Image.open(os.path.join(folder, f))))
                    ys.append(ci)
    return xs, ys


def _synthetic(per_class: int, offset: int):
    xs, ys = [], []
    for img, lab in synthetic.batch(CLASSES, per_class, offset):
        xs.append(to_array(img))
        ys.append(CLASSES.index(lab))
    return xs, ys


def _augment(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    xs, ys = _synthetic(args.per_class, TRAIN_SEED)
    rx, ry = _real_images()
    if rx:
        print(f"adding {len(rx)} real photos from data/train/ (x3 weight)")
        xs += rx * 3
        ys += ry * 3
    vx, vy = _synthetic(150, VAL_SEED)
    X, Y = torch.tensor(np.stack(xs)), torch.tensor(ys)
    VX, VY = torch.tensor(np.stack(vx)), torch.tensor(vy)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds also report "cuda"
    model = DefectCNN(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(X) + 63) // 64))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb = torch.stack([_augment(X[j]) for j in idx]).to(device)
            yb = Y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(VX.to(device)).argmax(1).cpu() == VY).float().mean().item()
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {total / len(X):.4f}  val_acc {acc:.3f}")

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    torch.save(model.cpu().state_dict(), WEIGHTS)
    with open(META, "w") as f:
        json.dump({"classes": CLASSES, "val_accuracy": round(acc, 4), "train_images": len(X), "real_photos": len(rx),
                   "trained_on": "synthetic + real photos" if rx else "synthetic only"}, f, indent=2)
    print(f"saved {WEIGHTS}")


if __name__ == "__main__":
    main()
