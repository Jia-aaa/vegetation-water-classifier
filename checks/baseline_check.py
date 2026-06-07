"""
多数类基线诊断：单独打印各类占比、多数类基线 accuracy，
并和 outputs/metrics.json 中的两次实验做对比。

这是任务文档明确要求的"先算基线"那一步。
"""
from pathlib import Path
import json
import sys

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "labels.tif"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

CLASS_NAMES = {0: "other", 1: "vegetation", 2: "water"}


def main():
    with rasterio.open(LABELS_PATH) as src:
        labels = src.read(1)

    classes, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum())

    print("=== Class distribution (whole image) ===")
    for c, n in zip(classes, counts):
        c = int(c)
        print(
            f"  {c} ({CLASS_NAMES.get(c, f'class_{c}')}): "
            f"{n} px, {n/total*100:.2f}%"
        )

    majority_class = int(classes[np.argmax(counts)])
    majority_ratio = float(counts.max() / total)
    print(
        f"\nmajority class: {majority_class} ({CLASS_NAMES[majority_class]})"
    )
    print(f"majority-class baseline accuracy (always predict majority): {majority_ratio:.4f}")

    if not METRICS_PATH.exists():
        print(f"\n(metrics.json not found at {METRICS_PATH}, run pipeline first.)")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    print("\n=== Lift over majority baseline ===")
    for split_name in ("random", "block"):
        s = m["splits"][split_name]
        acc = s["metrics"]["accuracy"]
        base = s["majority_baseline"]["accuracy"]
        lift = acc - base
        print(
            f"  {split_name:<7s} : acc={acc:.4f}  baseline={base:.4f}  lift={lift:+.4f}"
        )
        if lift < 0.02:
            print(f"    WARNING: model barely beats baseline on '{split_name}' split.")
        if s["metrics"]["per_class"]["2"]["recall"] < 0.5:
            print(
                f"    WARNING: water recall = "
                f"{s['metrics']['per_class']['2']['recall']:.2f} on '{split_name}' "
                f"split. The minority class is being missed."
            )


if __name__ == "__main__":
    main()
