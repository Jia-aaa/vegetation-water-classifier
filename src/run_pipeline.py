"""
主流水线：

1. 读 scene.tif + labels.tif，构造 7 维特征：
   4 个原始波段 Red/NIR/Green/SWIR + NDVI + NDWI + MNDWI。
2. 用两种切分跑同一个 RandomForest：
     - random : 像素随机 60/40
     - block  : 32x32 空间区块，60/40 按 block 整体切
3. 在测试集上算 accuracy / balanced accuracy / 混淆矩阵 / per-class P/R/F1 / macro F1。
4. 分别用 random 模型和 block 模型给整景做预测，
   输出 prediction_random.tif / prediction_spatial.tif + 三栏可视化。
5. 报告写入 outputs/metrics.json + outputs/report.txt。
"""
from pathlib import Path
import json
import sys

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import load_dataset, FEATURE_NAMES
from splits import split_random, split_block
from evaluation import majority_baseline, evaluate, format_report, CLASS_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SCENE_PATH = DATA_DIR / "scene.tif"
LABELS_PATH = DATA_DIR / "labels.tif"

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)


def train_and_eval(X, y, train_idx, test_idx, name):
    print(f"\n--- training: {name} ---")
    print(f"train pixels: {len(train_idx)}, test pixels: {len(test_idx)}")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    metrics = evaluate(y[test_idx], y_pred)
    baseline = majority_baseline(y[train_idx], y[test_idx])
    return clf, metrics, baseline


def class_distribution(y):
    classes, counts = np.unique(y, return_counts=True)
    total = int(counts.sum())
    return {
        int(c): {
            "name": CLASS_NAMES.get(int(c), f"class_{int(c)}"),
            "count": int(n),
            "ratio": float(n / total),
        }
        for c, n in zip(classes, counts)
    }


def save_prediction_raster(pred_flat, profile, shape, out_path):
    pred_img = pred_flat.reshape(shape).astype("uint8")
    out_profile = profile.copy()
    out_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 255,
            "compress": "lzw",
        }
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(pred_img, 1)
        dst.set_band_description(1, "prediction (0=other, 1=veg, 2=water)")


def save_visualization(pred_random, pred_block, labels, shape, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#bdbdbd", "#2ca02c", "#1f77b4"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(labels, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")
    axes[1].imshow(pred_random.reshape(shape), cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("Prediction (random split model)")
    axes[1].axis("off")
    im = axes[2].imshow(pred_block.reshape(shape), cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title("Prediction (block split model)")
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2], shrink=0.6)
    cbar.ax.set_yticklabels(["other", "vegetation", "water"])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def main():
    print("=== Vegetation / Water Classifier ===")

    if not SCENE_PATH.exists() or not LABELS_PATH.exists():
        print(f"ERROR: missing input file ({SCENE_PATH} / {LABELS_PATH}).")
        print("Place scene.tif and labels.tif into data/ before running.")
        sys.exit(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(SCENE_PATH, LABELS_PATH)
    X, y, shape, profile = ds["X"], ds["y"], ds["shape"], ds["profile"]
    print(f"feature names: {FEATURE_NAMES}")
    print(f"X shape: {X.shape}, y shape: {y.shape}, image shape: {shape}")

    dist = class_distribution(y)
    print("\nclass distribution (whole image):")
    for c, info in dist.items():
        print(f"  {c} ({info['name']}): {info['count']} px, {info['ratio']*100:.2f}%")

    n = X.shape[0]

    train_r, test_r = split_random(n)
    clf_r, m_r, b_r = train_and_eval(X, y, train_r, test_r, "random split")

    train_b, test_b = split_block(shape)
    clf_b, m_b, b_b = train_and_eval(X, y, train_b, test_b, "block split (32x32)")

    pred_full_random = clf_r.predict(X).astype("uint8")
    pred_full_block = clf_b.predict(X).astype("uint8")

    pred_random_path = OUTPUT_DIR / "prediction_random.tif"
    pred_spatial_path = OUTPUT_DIR / "prediction_spatial.tif"
    save_prediction_raster(pred_full_random, profile, shape, pred_random_path)
    save_prediction_raster(pred_full_block, profile, shape, pred_spatial_path)
    print(f"\nprediction (random split) raster: {pred_random_path}")
    print(f"prediction (block split) raster : {pred_spatial_path}")

    vis_path = OUTPUT_DIR / "preview.png"
    with rasterio.open(LABELS_PATH) as src:
        labels_2d = src.read(1)
    save_visualization(pred_full_random, pred_full_block, labels_2d, shape, vis_path)
    print(f"prediction preview: {vis_path}")

    metrics = {
        "feature_names": FEATURE_NAMES,
        "image_shape": list(shape),
        "class_distribution": {
            str(k): v for k, v in dist.items()
        },
        "model": "RandomForestClassifier",
        "model_params": {k: (v if not callable(v) else str(v)) for k, v in RF_PARAMS.items()},
        "splits": {
            "random": {
                "train_size": int(len(train_r)),
                "test_size": int(len(test_r)),
                "majority_baseline": b_r,
                "metrics": m_r,
            },
            "block": {
                "block_size": 32,
                "train_size": int(len(train_b)),
                "test_size": int(len(test_b)),
                "majority_baseline": b_b,
                "metrics": m_b,
            },
        },
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"metrics: {metrics_path}")

    report_lines = []
    report_lines.append("=== Class distribution (whole image) ===")
    for c, info in dist.items():
        report_lines.append(
            f"  {c} ({info['name']}): {info['count']} px, {info['ratio']*100:.2f}%"
        )
    report_lines.append("")
    report_lines.append(format_report("Random split (60/40 over pixels)", m_r, b_r))
    report_lines.append("")
    report_lines.append(format_report("Block split (32x32 blocks, 60/40 over blocks)", m_b, b_b))
    report_lines.append("")
    delta_acc = m_r["accuracy"] - m_b["accuracy"]
    delta_water_recall = (
        m_r["per_class"][2]["recall"] - m_b["per_class"][2]["recall"]
    )
    report_lines.append("=== Leakage diagnostic ===")
    report_lines.append(
        f"random_acc - block_acc = {delta_acc:+.4f}  "
        f"(positive = random split is optimistic / leaking)"
    )
    report_lines.append(
        f"random_water_recall - block_water_recall = {delta_water_recall:+.4f}"
    )

    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"report: {report_path}")

    print("\n" + "\n".join(report_lines))
    print("\n=== Pipeline finished ===")


if __name__ == "__main__":
    main()
