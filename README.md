# 植被 / 水体分类工具

基于 4 波段卫星影像（Red / NIR / Green / SWIR）和地面真值标签，
训练一个像素级的随机森林分类器（`0=其他 / 1=植被 / 2=水体`），
并提供严格的评估，避免被准确率欺骗。

> 任务核心：**当你没法肉眼判断对错时，怎么相信一个模型。**
> 见 [`EVALUATION.md`](EVALUATION.md)。

## 输入

- `data/scene.tif`：4 波段 GeoTIFF，descriptions = `('Red', 'NIR', 'Green', 'SWIR')`
- `data/labels.tif`：1 波段 uint8 GeoTIFF，`0=其他，1=植被，2=水体`

两者 CRS / transform / shape 必须一致。

## 输出（写入 `outputs/`）

| 文件                              | 内容                                               |
| --------------------------------- | -------------------------------------------------- |
| `outputs/prediction.tif`          | 整景预测（uint8，与输入同地理参考）                |
| `outputs/prediction_preview.png`  | 真值 vs 预测左右对比图                             |
| `outputs/metrics.json`            | 两种切分下的全部指标                               |
| `outputs/report.txt`              | 人读评估报告                                       |

## 安装

需要 Python 3.10 及以上：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python src/run_pipeline.py
```

流水线会：

1. 读 `data/scene.tif` + `data/labels.tif`，构造 6 维特征（4 波段 + NDVI + NDWI）
2. 用两种切分（**random** 像素随机 60/40，**block** 32×32 棋盘格 60/40）
   分别训练同一个 RF
3. 在测试集上算 accuracy / 混淆矩阵 / per-class P/R/F1 / macro F1
4. 用 block 切分的模型给整景做预测，存成 `prediction.tif` + 预览图
5. 把指标写到 `outputs/metrics.json` 和 `outputs/report.txt`

## 验证

`checks/` 下三个脚本各管一项任务硬性要求，可单独运行：

```bash
python checks/baseline_check.py        # 类别分布 + 多数类基线 + lift
python checks/leakage_diagnostic.py    # 用弱特征演示数据泄漏 / accuracy 假象
python checks/spatial_coherence.py     # 预测与真值的连通组件 / 碎片度
```

完整评估见 [`EVALUATION.md`](EVALUATION.md)。规格见 [`SPEC.md`](SPEC.md)。

## 项目结构

```
vegetation-water-classifier/
├── SPEC.md                          规格
├── EVALUATION.md                    评估报告（含数据泄漏诊断与解读）
├── README.md
├── requirements.txt
├── data/
│   ├── scene.tif                    输入影像（不入库）
│   └── labels.tif                   真值标签（不入库）
├── outputs/                         运行产物（不入库）
├── src/
│   ├── features.py                  6 维特征构造
│   ├── splits.py                    random / block 两种切分
│   ├── evaluation.py                指标 + 报告格式化
│   └── run_pipeline.py              主流水线入口
└── checks/
    ├── baseline_check.py
    ├── leakage_diagnostic.py
    └── spatial_coherence.py
```
