"""
模型评估：算多数类基线、accuracy、混淆矩阵、per-class precision/recall/F1、macro F1，
统一返回一个 dict 方便序列化到 JSON / 报告中。
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "other", 1: "vegetation", 2: "water"}


def majority_baseline(y_train, y_test):
    """全猜训练集中最常见的那一类，得到的 accuracy。"""
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = int(classes[np.argmax(counts)])
    pred = np.full_like(y_test, majority_class)
    return {
        "majority_class": majority_class,
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def evaluate(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    per_class = {}
    for i, c in enumerate(CLASS_LABELS):
        per_class[c] = {
            "name": CLASS_NAMES[c],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_LABELS,
        "per_class": per_class,
    }


def format_report(name, metrics, baseline=None):
    lines = [f"=== {name} ===", f"accuracy          : {metrics['accuracy']:.4f}"]
    if baseline is not None:
        lines.append(
            f"majority-class baseline acc: {baseline['accuracy']:.4f}  "
            f"(class {baseline['majority_class']} = {CLASS_NAMES[baseline['majority_class']]})"
        )
        lift = metrics["accuracy"] - baseline["accuracy"]
        lines.append(f"lift over baseline: {lift:+.4f}")
    lines.append(f"balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    lines.append(f"macro F1          : {metrics['macro_f1']:.4f}")
    lines.append("confusion matrix (rows=true, cols=pred):")
    cm = np.array(metrics["confusion_matrix"])
    header = "              " + " ".join(f"{CLASS_NAMES[c]:>10}" for c in CLASS_LABELS)
    lines.append(header)
    for i, c in enumerate(CLASS_LABELS):
        row = " ".join(f"{v:>10d}" for v in cm[i])
        lines.append(f"  {CLASS_NAMES[c]:<10}  {row}")
    lines.append("per-class precision / recall / F1:")
    for c in CLASS_LABELS:
        info = metrics["per_class"][c]
        lines.append(
            f"  {info['name']:<10}  P={info['precision']:.4f}  "
            f"R={info['recall']:.4f}  F1={info['f1']:.4f}  n={info['support']}"
        )
    return "\n".join(lines)
