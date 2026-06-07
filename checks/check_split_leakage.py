"""
切分泄漏诊断（任务核心 P0）。

任务原文："你怎么保证你的测试集没在偷看训练集？"
本脚本用三个硬指标回答：

  1. block 切分：train_blocks ∩ test_blocks = ∅
     （证明它真的是 block 级切分，不是把同一个 block 的像素再随机分）

  2. random 切分：测量 test 像素到最近 train 像素的 4-邻域距离分布
     （≤1 表示测试像素紧贴训练像素，是空间近邻泄漏的根源）

  3. block 切分：同样测量距离分布
     （block split 只保证同一 block 不同时进入 train/test；
      相邻 block 一个分到 train、一个分到 test 时，边界两侧像素仍可能挨着，
      所以这个分布是**残余近邻风险**的量化，而非"完全分离"的证明）

结果以表格形式打印；如果 block 切分发现任何 block 同时出现在
train 与 test 中，直接报错退出。
"""
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import load_dataset
from splits import split_random, split_block, BLOCK_SIZE


def block_id_grid(shape, block_size):
    h, w = shape
    rows, cols = np.indices((h, w))
    return (rows // block_size) * 10000 + (cols // block_size)


def distance_distribution(train_idx, test_idx, shape):
    """
    把 train 像素标 1，test 像素标 0，对 train mask 做欧氏距离变换；
    再在 test 像素位置取距离值，得到每个 test 像素到最近 train 像素的距离。
    """
    h, w = shape
    train_mask = np.zeros(h * w, dtype=bool)
    train_mask[train_idx] = True
    train_mask_2d = train_mask.reshape(h, w)

    not_train = ~train_mask_2d
    dist = distance_transform_edt(not_train)

    test_mask = np.zeros(h * w, dtype=bool)
    test_mask[test_idx] = True
    test_mask_2d = test_mask.reshape(h, w)

    test_distances = dist[test_mask_2d]
    return test_distances


def summarize(name, distances):
    print(
        f"  {name:<28s}  "
        f"min={distances.min():.2f}  "
        f"median={np.median(distances):.2f}  "
        f"mean={distances.mean():.2f}  "
        f"max={distances.max():.2f}  "
        f"frac<=1={(distances <= 1).mean()*100:.2f}%"
    )


def main():
    print("=== Split leakage diagnostic ===\n")
    ds = load_dataset(ROOT / "data" / "scene.tif", ROOT / "data" / "labels.tif")
    shape = ds["shape"]
    n = ds["X"].shape[0]
    print(f"image shape : {shape}, total pixels : {n}")
    print(f"block size  : {BLOCK_SIZE}\n")

    train_r, test_r = split_random(n)
    train_b, test_b = split_block(shape)

    bgrid = block_id_grid(shape, BLOCK_SIZE).ravel()
    train_blocks = set(np.unique(bgrid[train_b]).tolist())
    test_blocks = set(np.unique(bgrid[test_b]).tolist())
    overlap = train_blocks & test_blocks

    print("=== Block disjointness (block split) ===")
    print(f"  unique train blocks : {len(train_blocks)}")
    print(f"  unique test  blocks : {len(test_blocks)}")
    print(f"  overlap             : {len(overlap)}")
    if overlap:
        print(f"FAIL: blocks shared between train and test: {sorted(overlap)[:10]}...")
        sys.exit(1)
    print("  OK: train_blocks AND test_blocks are disjoint (intersection is empty)\n")

    print("=== Pixel overlap (sanity) ===")
    pix_overlap_random = len(set(train_r.tolist()) & set(test_r.tolist()))
    pix_overlap_block = len(set(train_b.tolist()) & set(test_b.tolist()))
    print(f"  random split: |train AND test| = {pix_overlap_random} pixels")
    print(f"  block  split: |train AND test| = {pix_overlap_block} pixels")
    if pix_overlap_random or pix_overlap_block:
        print("FAIL: pixel-level overlap detected.")
        sys.exit(1)
    print("  OK: no pixel-level overlap in either split\n")

    print("=== Distance from test pixel to nearest train pixel ===")
    print("  (4/8-neighbour-ish Euclidean distance, in pixel units)")
    d_random = distance_distribution(train_r, test_r, shape)
    d_block = distance_distribution(train_b, test_b, shape)
    summarize("random split", d_random)
    summarize("block split (32x32)", d_block)

    print("\n=== Interpretation ===")
    frac_close_random = (d_random <= 1).mean()
    frac_close_block = (d_block <= 1).mean()
    print(
        f"  random split: {frac_close_random*100:.2f}% of test pixels have a "
        f"train pixel within 1 px -- i.e. test is sitting on top of train."
    )
    print(
        f"  block  split: {frac_close_block*100:.2f}% of test pixels have a "
        f"train pixel within 1 px -- residual near-neighbour risk at block boundaries."
    )
    print(
        "  => block split avoids same-block leakage and substantially reduces "
        "the near-neighbour leakage of random split, but it does not fully "
        "eliminate spatial autocorrelation."
    )


if __name__ == "__main__":
    main()
