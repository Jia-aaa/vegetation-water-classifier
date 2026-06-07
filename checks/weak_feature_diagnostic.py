"""
弱特征诊断（"高 accuracy 假象"对照实验）。

任务原文："你要能用一句话说清：为什么一个准确率 99% 的模型，可能其实毫无用处。"

主流水线在 7 维特征下两种切分都到 1.0，看不出单类失败。
本脚本故意把模型弱化为只用 1 个特征 (Red)，让小类（水体）失败暴露：
  - 总体 accuracy 仍能维持 ~91%（被多数类抬起来）
  - water recall 跌到 ~10% 以下
  - 这就是任务要警惕的"高 acc 掩盖小类失败"的活例子

注：检查 train/test 是否真的空间分离，请看 check_split_leakage.py。
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
