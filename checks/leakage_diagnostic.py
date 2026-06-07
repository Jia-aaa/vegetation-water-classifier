"""
切分泄漏诊断：

主流水线用 4 波段 + NDVI + NDWI 时，类别在特征空间里完全可分，
两种切分的 accuracy 都达到 1.0，看不出 random / block 的差异。

为了演示"切分方法学很重要"这一点，本脚本故意把模型弱化：
只用 1 个特征 (Red) 训练 RandomForest。这种情况下，random 切分
会因为相邻像素几乎相同而虚高，block 切分会显著下降。

如果 random_acc - block_acc 显著为正，就证明：
  - 我们的 block 切分确实切断了空间相邻；
  - "随机切分高 / 区块切分低"是数据泄漏的典型模式。
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import load_dataset
from splits import split_random, split_block
from evaluation import evaluate, format_report, majority_baseline


WEAK_FEATURE_INDEX = 0
WEAK_FEATURE_NAME = "Red only"


def main():
    ds = load_dataset(ROOT / "data" / "scene.tif", ROOT / "data" / "labels.tif")
    X, y, shape = ds["X"], ds["y"], ds["shape"]

    Xw = X[:, [WEAK_FEATURE_INDEX]]
    print(f"=== Leakage diagnostic with weak features: {WEAK_FEATURE_NAME} ===")
    print(f"X shape: {Xw.shape}")

    rf_kwargs = dict(
        n_estimators=200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    train_r, test_r = split_random(len(y))
    clf_r = RandomForestClassifier(**rf_kwargs).fit(Xw[train_r], y[train_r])
    m_r = evaluate(y[test_r], clf_r.predict(Xw[test_r]))
    b_r = majority_baseline(y[train_r], y[test_r])

    train_b, test_b = split_block(shape)
    clf_b = RandomForestClassifier(**rf_kwargs).fit(Xw[train_b], y[train_b])
    m_b = evaluate(y[test_b], clf_b.predict(Xw[test_b]))
    b_b = majority_baseline(y[train_b], y[test_b])

    print()
    print(format_report("Weak feats + Random split", m_r, b_r))
    print()
    print(format_report("Weak feats + Block split (32x32)", m_b, b_b))
    print()

    delta = m_r["accuracy"] - m_b["accuracy"]
    print(f"random_acc - block_acc = {delta:+.4f}")
    if delta > 0.02:
        print("=> CONFIRMED: random split is optimistic, block split is honest.")
        print("   This is the data-leakage pattern the task warns about.")
    else:
        print("=> No significant gap on this weak baseline.")


if __name__ == "__main__":
    main()
