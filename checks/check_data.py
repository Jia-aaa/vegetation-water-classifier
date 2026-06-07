"""
输入数据 sanity 检查（任务硬性要求："去统计一下 各类占比"）：

  1. data/scene.tif 与 data/labels.tif 都存在
  2. scene 是 4 波段 float（任务给定的 Red/NIR/Green/SWIR）
  3. labels 是 1 波段 uint8
  4. shape / CRS / transform 一致
  5. labels 取值 ⊂ {0, 1, 2}（没有未声明的 nodata 值混进来）
  6. 打印各类占比与多数类基线 accuracy
"""
from pathlib import Path
import sys

import numpy as np
import rasterio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "data" / "scene.tif"
LABELS_PATH = ROOT / "data" / "labels.tif"

CLASS_NAMES = {0: "other", 1: "vegetation", 2: "water"}
ALLOWED_LABELS = {0, 1, 2}


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"OK  : {msg}")


def main():
    print("=== Input data sanity checks ===")

    if not SCENE_PATH.exists():
        fail(f"missing {SCENE_PATH}")
    if not LABELS_PATH.exists():
        fail(f"missing {LABELS_PATH}")
    ok("both scene.tif and labels.tif exist")

    with rasterio.open(SCENE_PATH) as src:
        scene_count = src.count
        scene_shape = (src.height, src.width)
        scene_crs = src.crs
        scene_transform = src.transform
        scene_dtypes = src.dtypes
        scene_descs = src.descriptions

    with rasterio.open(LABELS_PATH) as src:
        labels_count = src.count
        labels_shape = (src.height, src.width)
        labels_crs = src.crs
        labels_transform = src.transform
        labels_dtype = src.dtypes[0]
        labels = src.read(1)

    if scene_count != 4:
        fail(f"scene must be 4 bands, got {scene_count}")
    ok(f"scene has 4 bands  (descriptions={scene_descs}, dtypes={scene_dtypes})")

    if labels_count != 1:
        fail(f"labels must be 1 band, got {labels_count}")
    ok(f"labels has 1 band  (dtype={labels_dtype})")

    if scene_shape != labels_shape:
        fail(f"shape mismatch: scene={scene_shape}, labels={labels_shape}")
    ok(f"scene and labels share shape {scene_shape}")

    if scene_crs != labels_crs:
        fail(f"CRS mismatch: scene={scene_crs}, labels={labels_crs}")
    ok(f"CRS matches: {scene_crs}")

    if scene_transform != labels_transform:
        fail("transform mismatch between scene and labels")
    ok("transform matches")

    unique_labels = set(int(v) for v in np.unique(labels))
    extras = unique_labels - ALLOWED_LABELS
    if extras:
        fail(f"labels contain unexpected values {sorted(extras)}; expected subset of {sorted(ALLOWED_LABELS)}")
    ok(f"labels values are subset of {{0,1,2}}: actually {sorted(unique_labels)}")

    print("\n=== Class distribution ===")
    classes, counts = np.unique(labels, return_counts=True)
    total = int(counts.sum())
    for c, n in zip(classes, counts):
        c = int(c)
        print(
            f"  {c} ({CLASS_NAMES.get(c, f'class_{c}')}): "
            f"{n} px, {n/total*100:.2f}%"
        )
    majority_class = int(classes[np.argmax(counts)])
    majority_acc = float(counts.max() / total)
    print(
        f"\nmajority class = {majority_class} "
        f"({CLASS_NAMES[majority_class]}); baseline accuracy = {majority_acc:.4f}"
    )
    print("\nAll input checks PASSED.")


if __name__ == "__main__":
    main()
