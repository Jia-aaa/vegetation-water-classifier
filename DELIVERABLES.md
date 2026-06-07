# 📦 交付清单

> 本文件唯一目的：**让导师 30 秒内对上任务要的东西在哪。**

---

## 任务原文（TASK_2.md 第三步）

> 交**规格**、**代码**、以及一份**评估报告**：
> 多数类基线、混淆矩阵、各类 precision/recall、两种切分方式下的准确率对比。
> 导师会重点和你讨论第 3 条（数据泄漏）。
>
> 这个任务做完，你要能用一句话说清：**为什么一个准确率 99% 的模型，可能其实毫无用处。**

---

## 1. 规格 → [`SPEC.md`](SPEC.md)

包含：目标、输入统计、特征、模型选择、训练/测试切分（random vs block）、评估指标、验收标准、非目标。

## 2. 代码

| 模块 | 路径 | 说明 |
|---|---|---|
| 主流水线 | [`src/run_pipeline.py`](src/run_pipeline.py) | 一条命令跑完：特征 → 两种切分训练 → 指标 → 整景预测 → 可视化 |
| 特征构造 | [`src/features.py`](src/features.py) | 7 维：Red / NIR / Green / SWIR / NDVI / NDWI / MNDWI |
| 切分 | [`src/splits.py`](src/splits.py) | `split_random` / `split_block`（32×32 区块整体切） |
| 评估 | [`src/evaluation.py`](src/evaluation.py) | accuracy / balanced_acc / macro F1 / 混淆矩阵 / per-class P/R/F1 |
| 自检 | [`checks/`](checks/) | 5 个独立脚本（见下表） |

复现方式：

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

5 个自检脚本可单独运行：

| 脚本 | 作用 |
|---|---|
| `checks/check_data.py` | 输入 sanity：波段数、dtype、CRS、shape、标签值域 |
| `checks/check_split_leakage.py` | **P0**：实证 block 切分 train∩test=∅，量化两种切分的近邻距离分布 |
| `checks/baseline_check.py` | 多数类基线 + 模型 lift |
| `checks/weak_feature_diagnostic.py` | "99% 假象"对照：弱特征下水体 recall 暴跌 |
| `checks/spatial_coherence.py` | 预测与真值的连通组件 / 碎片度对比 |

## 3. 评估报告 → [`EVALUATION.md`](EVALUATION.md)

任务点名要的 4 项指标，在报告中的位置：

| 任务要求 | 在 EVALUATION.md 的位置 |
|---|---|
| 多数类基线 | 第 1 节 "类别分布与多数类基线" + 第 4 节 "多数类基线对比" |
| 混淆矩阵 | 第 3.2 节（强特征） + 第 3.3 节（弱特征）的 confusion matrix |
| 各类 precision / recall | 第 3.2 节 + 第 3.3 节的 per-class 表 |
| **两种切分方式下的准确率对比** | **第 3 节（整节都是）** —— 导师会和你讨论的就是这部分 |

第 3 节 "数据泄漏诊断" 是任务最看重的部分，包含：

- **3.1 切分方法学是否成立（结构层面）** —— 用 `check_split_leakage.py` 的硬数据回答
  "你怎么保证测试集没在偷看训练集"：random 切分下 97.28% 测试像素紧贴训练像素，
  block 切分 train_blocks ∩ test_blocks = ∅
- **3.2 在本数据集上两种切分的 accuracy 对比（结果层面）**
- **3.3 在弱特征下两种切分都暴露了"高 acc 假象"** —— water recall 只有 8–10%
- **3.4 random vs block 在本数据集 accuracy 差距很小的原因分析**

## 4. 一句话答辩 → [`EVALUATION.md` 第 7 节](EVALUATION.md)

一个 accuracy 99% 的模型可能毫无用处：
（a）遥感影像相邻像素高度相似——random 切分下 **97.28% 的测试像素紧挨训练像素**——模型只是在"识别近邻"而非泛化；
（b）多数类占比可能极高，全猜 vegetation 也能拿到 73% accuracy，弱特征下水体 recall 只有 8% 也能撑起 91% 的总体 accuracy。
判断模型是否可信，必须**同时看**：block 切分下的混淆矩阵 + 每一类的 precision/recall + balanced accuracy + 多数类基线 lift——缺一不可。

---

## 📂 一图看懂仓库结构

```
vegetation-water-classifier/
│
├── DELIVERABLES.md          ← 你正在看的这份（交付索引）
├── TASK_2.md                ← 任务原文
├── SPEC.md                  ← 【交付 1】规格
├── EVALUATION.md            ← 【交付 3】评估报告（含一句话答辩）
├── TASK_COMPLIANCE.md       ← 任务要求逐条对照
├── README.md
├── requirements.txt
│
├── src/                     ← 【交付 2a】主流水线代码
│   ├── run_pipeline.py
│   ├── features.py
│   ├── splits.py
│   └── evaluation.py
│
├── checks/                  ← 【交付 2b】独立自检脚本
│   ├── check_data.py
│   ├── check_split_leakage.py    ← 任务第 3 条核心
│   ├── baseline_check.py
│   ├── weak_feature_diagnostic.py
│   └── spatial_coherence.py
│
├── data/                    （不入库：scene.tif + labels.tif）
└── outputs/                 （不入库：metrics.json / report.txt / preview.png / 两张 prediction tif）
```
