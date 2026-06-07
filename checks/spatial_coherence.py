"""
空间合理性自检：

任务文档第 4 条要求"把预测结果作为一张图叠回原影像，看水体形状是否连成片"。
这里我们用一个量化指标 (而不是仅仅看图)：

- 计算 prediction.tif 中每一类的"碎片度"：
    fragmentation = (4-邻域连通组件数) / (该类像素数)
  值越低表示越成片，越高表示越像椒盐噪声。

- 同时与 labels.tif 中每一类的碎片度对比，
  作为一个相对参考 (预测不应该比真值碎得多)。
"""
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label as cc_label


ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "outputs" / "prediction.tif"
LABELS_PATH = ROOT / "data" / "labels.tif"

CLASS_NAMES = {0: "other", 1: "vegetation", 2: "water"}


def fragmentation(arr, cls):
    mask = arr == cls
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        return None
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, n_components = cc_label(mask, structure=structure)
    return {
        "pixels": n_pixels,
        "components": int(n_components),
        "fragmentation": float(n_components / n_pixels),
    }


def main():
    if not PRED_PATH.exists():
        print(f"ERROR: {PRED_PATH} not found, run pipeline first.")
        return
    with rasterio.open(PRED_PATH) as src:
        pred = src.read(1)
    with rasterio.open(LABELS_PATH) as src:
        labels = src.read(1)

    print("=== Spatial coherence (4-connectivity) ===")
    print(
        f"{'class':<12}{'truth_px':>10}{'truth_cc':>10}{'truth_frag':>12}"
        f"{'pred_px':>10}{'pred_cc':>10}{'pred_frag':>12}"
    )
    for c in (0, 1, 2):
        t = fragmentation(labels, c)
        p = fragmentation(pred, c)
        if t is None or p is None:
            continue
        print(
            f"{CLASS_NAMES[c]:<12}"
            f"{t['pixels']:>10d}{t['components']:>10d}{t['fragmentation']:>12.4f}"
            f"{p['pixels']:>10d}{p['components']:>10d}{p['fragmentation']:>12.4f}"
        )

    print(
        "\nNote: a much higher pred_frag than truth_frag for the same class "
        "would indicate salt-and-pepper noise."
    )


if __name__ == "__main__":
    main()
