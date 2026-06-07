# 植被 / 水体分类工具

基于 4 波段卫星影像（Red / NIR / Green / SWIR）和地面真值标签，
训练一个像素级的随机森林分类器（`0=其他 / 1=植被 / 2=水体`），
并提供严格的评估，避免被准确率欺骗。

## 任务重点

本项目不追求"高 accuracy"——追求的是**当你看不出对错时，怎么判断模型可不可信**。

遥感影像相邻像素高度相似（空间自相关），随机按像素切训练 / 测试会让测试集
紧贴训练集，accuracy 看起来很漂亮但模型没真正泛化。所以本项目同时跑两种切分：

1. **随机像素切分**——作为**对照**，演示泄漏的存在
2. **空间区块切分（32×32 棋盘格）**——作为**更可信**的泛化评估

最终结论以 block 切分 + 多数类基线 + per-class recall 共同判断。
完整评估见 [`EVALUATION.md`](EVALUATION.md)，任务对齐表见 [`TASK_COMPLIANCE.md`](TASK_COMPLIANCE.md)。

## 输入

- `data/scene.tif`：4 波段 GeoTIFF，descriptions = `('Red', 'NIR', 'Green', 'SWIR')`
- `data/labels.tif`：1 波段 uint8 GeoTIFF，`0=其他，1=植被，2=水体`

两者 CRS / transform / shape 必须一致。

## 输出（写入 `outputs/`）

| 文件                              | 内容                                               |
| --------------------------------- | -------------------------------------------------- |
| `outputs/prediction_random.tif`   | random 切分模型整景预测（uint8，同地理参考）        |
| `outputs/prediction_spatial.tif`  | block 切分模型整景预测（uint8，同地理参考）         |
| `outputs/preview.png`             | 真值 / random / spatial 三栏对比图                 |
| `outputs/metrics.json`            | 两种切分下的全部指标                                |
| `outputs/report.txt`              | 人读评估报告                                        |

## 安装与运行

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

流水线会：

1. 读 `data/scene.tif` + `data/labels.tif`，构造 7 维特征（4 波段 + NDVI + NDWI + MNDWI）
2. 用两种切分（random 像素 60/40，block 32×32 棋盘格 60/40）分别训练同一个 RF
3. 在测试集上算 accuracy / balanced_accuracy / macro_F1 / 混淆矩阵 / per-class P/R/F1
4. 用两个模型分别给整景做预测，存成 `prediction_random.tif` / `prediction_spatial.tif` + 三栏 preview
5. 把指标写到 `outputs/metrics.json` 和 `outputs/report.txt`

## 独立验证（`checks/`）

```bash
python checks/check_data.py              # 输入数据 sanity（4 波段、shape、labels∈{0,1,2}）
python checks/check_split_leakage.py     # P0: 实证 block 切分 train∩test=∅ + 像素距离分布
python checks/baseline_check.py          # 多数类基线 + lift
python checks/weak_feature_diagnostic.py # "99% 假象"对照：弱特征下水体 recall 暴跌
python checks/spatial_coherence.py       # 预测与真值的连通组件 / 碎片度
```

## 项目结构

```
vegetation-water-classifier/
├── SPEC.md                           规格
├── EVALUATION.md                     评估报告（含数据泄漏诊断与解读）
├── TASK_COMPLIANCE.md                逐条任务要求对照
├── README.md
├── requirements.txt
├── data/
│   ├── scene.tif                     输入影像（不入库）
│   └── labels.tif                    真值标签（不入库）
├── outputs/                          运行产物（不入库）
├── src/
│   ├── features.py                   7 维特征构造
│   ├── splits.py                     random / block 两种切分
│   ├── evaluation.py                 指标 + 报告格式化
│   └── run_pipeline.py               主流水线入口
└── checks/
    ├── check_data.py
    ├── check_split_leakage.py
    ├── baseline_check.py
    ├── weak_feature_diagnostic.py
    └── spatial_coherence.py
```

## 一句话答辩

一个 accuracy 99% 的模型可能毫无用处：
（a）随机切分会让测试像素紧挨训练像素，模型只是在"识别近邻"而非泛化；
（b）多数类占比可能极高，全猜多数类也能逼近 99%。
判断模型可不可信，必须看 **block 切分下的混淆矩阵 + 每类 precision/recall + balanced accuracy + 多数类基线 lift**，缺一不可。
