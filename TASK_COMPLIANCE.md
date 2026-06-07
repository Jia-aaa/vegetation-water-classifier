# 任务对齐表

> 本文件逐条对应 `TASK_2.md` 中的硬性要求，并指明本仓库哪些文件 / 命令满足该要求。
> 导师可以按表索引快速定位证据。

## 1. 规格先行

- 对应文件：[`SPEC.md`](SPEC.md)
- 内容：目标 / 输入（含类别占比）/ 方法（特征 + 模型）/ 切分（random + block，含为什么 block 更可信）/ 评估指标 / 验收标准 / 非目标。

## 2. 多数类基线

- 对应文件：[`src/run_pipeline.py`](src/run_pipeline.py)、[`checks/baseline_check.py`](checks/baseline_check.py)、[`EVALUATION.md`](EVALUATION.md) 第 1 节
- 做法：统计 `labels.tif` 中 0/1/2 三类像素占比；占比最高的类别作为多数类；记录 "全猜多数类" 的 accuracy。
- 实测：majority class = 1（vegetation），baseline accuracy ≈ **0.7340**。

## 3. 不只看 accuracy

- 对应文件：[`src/evaluation.py`](src/evaluation.py)、[`outputs/metrics.json`](outputs/metrics.json)、[`EVALUATION.md`](EVALUATION.md) 第 2 节
- 报告的指标（每种切分都有）：
  - overall accuracy
  - **balanced accuracy**（对类别不平衡更稳）
  - **macro F1**
  - **3×3 confusion matrix**
  - **per-class precision / recall / F1 / support**
  - lift over majority baseline

## 4. 数据泄漏诊断（任务最强调的一关）

任务原文："你怎么保证你的测试集没在偷看训练集？换一种按空间区块切分的方式，准确率会不会掉下来？把两种切分的结果都报出来。"

本仓库的回答分两层：

### 4.1 切分方法学是否成立

- 对应文件：[`checks/check_split_leakage.py`](checks/check_split_leakage.py)
- 用三个硬指标实证 block 切分确实空间分离：
  1. `train_blocks ∩ test_blocks = ∅`（block 级互不相交）
  2. train / test 像素集合互不相交
  3. test 像素到最近 train 像素的距离分布：random 切分下 ~96% 的 test 像素紧贴 train 像素（≤1 px），block 切分下只有 block 边界上的少量像素紧贴
- 结论：block 切分**结构上不可能泄漏**；random 切分**结构上一定泄漏**。

### 4.2 在本数据集上两种切分结果对比

- 对应文件：[`EVALUATION.md`](EVALUATION.md) 第 3 节
- 主结果（7 维特征）：两种切分都到 1.0，类别在特征空间完全可分，看不出 gap——这是数据本身性质，不是 bug。
- 弱特征对照（仅 Red，[`checks/weak_feature_diagnostic.py`](checks/weak_feature_diagnostic.py)）：
  - random / block 的 overall accuracy 都 ≈ 0.91
  - 但 **water recall ≈ 0.08–0.10**
  - 这正是任务结尾那句"99% 准确率却毫无用处"的活例子——多数类抬高了 accuracy，把小类失败完全藏起来

## 5. 空间合理性

- 对应文件：[`outputs/prediction_random.tif`](outputs/prediction_random.tif)、[`outputs/prediction_spatial.tif`](outputs/prediction_spatial.tif)、[`outputs/preview.png`](outputs/preview.png)、[`checks/spatial_coherence.py`](checks/spatial_coherence.py)
- 量化指标：连通组件数 + 4-邻域碎片度（fragmentation = components / pixels）
- 实测：预测结果在每一类上的 components 数与 truth 完全一致，无椒盐噪声。

## 6. 一句话答辩

一个 accuracy 99% 的模型可能毫无用处，原因有二：
（a）遥感影像相邻像素高度相似，随机按像素切分会让测试集里大量像素紧挨着训练像素，模型只是在"识别近邻"，并没有泛化到新区域；
（b）多数类占比可能极高，全猜多数类也能拿到接近 99% 的 accuracy。
所以判断模型是否可信，必须同时看 **空间区块切分下** 的 **混淆矩阵 + 每一类的 precision/recall + balanced accuracy + 多数类基线 lift**，缺一不可。

## 7. 复现与产物清单

```bash
pip install -r requirements.txt
python src/run_pipeline.py            # 主流水线
python checks/check_data.py           # 输入检查
python checks/check_split_leakage.py  # 切分泄漏诊断 (P0)
python checks/baseline_check.py       # 多数类基线
python checks/weak_feature_diagnostic.py  # "99% 假象"对照实验
python checks/spatial_coherence.py    # 空间合理性
```

产物：

```
outputs/
  prediction_random.tif    # 随机切分模型在整景上的预测
  prediction_spatial.tif   # block 切分模型在整景上的预测
  preview.png              # 真值 / random / spatial 三栏对比
  metrics.json             # 全量指标 (含 balanced accuracy / macro F1 / per-class)
  report.txt               # 人读评估报告
```
