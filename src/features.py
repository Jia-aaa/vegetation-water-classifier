"""
特征工程：从 scene.tif 读 4 波段，派生 NDVI / NDWI / MNDWI，
生成像素级特征矩阵 X (7 维) 和标签向量 y。

返回的 X / y 是按行优先 (row-major) 展开的，
形状分别是 (H*W, n_features) 与 (H*W,)，
顺序与 numpy 的 .ravel() 一致。
"""
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "data" / "scene.tif"
LABELS_PATH = ROOT / "data" / "labels.tif"


FEATURE_NAMES = ["Red", "NIR", "Green", "SWIR", "NDVI", "NDWI", "MNDWI"]


def safe_div(num, den):
    out = np.zeros_like(num, dtype="float32")
    valid = den != 0
    out[valid] = num[valid] / den[valid]
    return out


def build_features(scene):
    red = scene[0].astype("float32")
    nir = scene[1].astype("float32")
    green = scene[2].astype("float32")
    swir = scene[3].astype("float32")

    ndvi = safe_div(nir - red, nir + red)
    ndwi = safe_div(green - nir, green + nir)
    mndwi = safe_div(green - swir, green + swir)

    stack = np.stack([red, nir, green, swir, ndvi, ndwi, mndwi], axis=0)
    return stack


def load_dataset(scene_path=SCENE_PATH, labels_path=LABELS_PATH):
    with rasterio.open(scene_path) as src:
        scene = src.read()
        profile = src.profile
        height, width = src.height, src.width

    with rasterio.open(labels_path) as src:
        labels = src.read(1)

    if labels.shape != (height, width):
        raise ValueError(
            f"labels shape {labels.shape} != scene shape {(height, width)}"
        )

    feats = build_features(scene)
    n_feat = feats.shape[0]

    X = feats.reshape(n_feat, height * width).T.astype("float32")
    y = labels.reshape(height * width).astype("int64")

    return {
        "X": X,
        "y": y,
        "shape": (height, width),
        "profile": profile,
        "feature_names": FEATURE_NAMES,
    }
