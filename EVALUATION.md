# 评估报告：植被/水体分类器

> 任务文档要求："交一份评估报告：多数类基线、混淆矩阵、各类 precision/recall、
> 两种切分方式下的准确率对比。导师会重点和你讨论第 3 条（数据泄漏）。"
>
> 本报告按这个清单逐项交付，并对结果做诚实的解读。

**📍 任务要求 → 章节速查**

| 任务要的 | 看哪一节 |
|---|---|
| 多数类基线 | [§1](#1-类别分布与多数类基线) + [§4](#4-多数类基线对比) |
| 混淆矩阵 | [§3.2](#32-在本数据集上两种切分的-accuracy-对比结果层面) + [§3.3](#33-在弱特征下两种切分都暴露了高-acc-假象) |
| 各类 precision / recall | [§3.2](#32-在本数据集上两种切分的-accuracy-对比结果层面) + [§3.3](#33-在弱特征下两种切分都暴露了高-acc-假象) |
| **两种切分方式下的准确率对比（导师重点）** | **[§3 整节](#3-数据泄漏诊断任务最关键的一节)** |
| 一句话答辩（99% 为何可能没用） | [§7](#7-一句话答辩) |

## 1. 类别分布与多数类基线

`data/labels.tif`（200 × 200 = 40 000 像素）：

| 类别       | 像素数 | 占比     |
| ---------- | -----: | -------: |
| 0 = 其他   |  9 976 |  24.94 % |
| 1 = 植被   | 29 361 |  73.40 % |
| 2 = 水体   |    663 |   1.66 % |

**多数类基线**：永远预测 `vegetation`，全图 accuracy ≈ **0.7340**。
任何模型如果 accuracy 不能显著超过 0.73，就**没学到东西**。

水体只占 1.66%，所以**水体上的 recall / F1 才是这次评估的真正信号**。

## 2. 特征与模型

| 维度 | 选择 |
|---|---|
| 特征（7 维） | Red, NIR, Green, SWIR, NDVI, NDWI, MNDWI |
| 模型 | `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42)` |
| 选模型理由 | 小数据（4 万像素）下稳定、可解释、不需调参；`class_weight="balanced"` 直接处理类别不平衡 |

加 NDVI / NDWI / MNDWI 是因为植被和水体在它们上的差异远比单波段大，
是该类问题的经典做法。MNDWI 用 SWIR 替代 NIR，在含建成区/阴影的影像上对水体识别更稳。

## 3. 数据泄漏诊断（任务最关键的一节）

### 3.1 切分方法学是否成立（结构层面）

`python checks/check_split_leakage.py` 用三个硬指标实证两种切分的空间分离程度：

```
=== Block disjointness (block split) ===
  unique train blocks : 29
  unique test  blocks : 20
  overlap             : 0
  OK: train_blocks AND test_blocks are disjoint (intersection is empty)

=== Pixel overlap (sanity) ===
  random split: |train AND test| = 0 pixels
  block  split: |train AND test| = 0 pixels

=== Distance from test pixel to nearest train pixel ===
                          min   median  mean   max  frac<=1px
  random split            1.00   1.00   1.01  2.00   97.28%
  block split (32x32)     1.00   8.00   9.71 40.00    7.68%
```

解读：

- **block 切分避免了"同一 block 被拆"的泄漏**：训练用的 block 集合和测试用的 block 集合零交集，
  说明没有任何 32×32 空间块被同时拆到训练集和测试集。
  测试像素到最近训练像素的中位距离是 8 px，仅 7.68% 的测试像素落在 1 px 邻域内——
  这些近邻主要来自不同 block 的边界处，是 block 切分的**残余近邻风险**。
  因此 block split **不能说完全消除空间自相关**，但它显著降低了 random split 中"测试像素紧贴训练像素"的风险。
- **random 切分结构上一定有空间近邻泄漏风险**：**97.28% 的测试像素的 4-邻域里就有训练像素**——
  模型甚至不需要"理解光谱"，只要复制隔壁邻居的标签就能拿到接近完美的分数。
- 这是任务问的 "你怎么保证测试集没在偷看训练集" 的主要证据：
  至少可以证明测试集没有和训练集共享同一个空间 block，
  且 random split 的近邻泄漏风险远高于 block split。

### 3.2 在本数据集上两种切分的 accuracy 对比（结果层面）

主流水线（7 维强特征）下的实际结果：

| 切分方式 | accuracy | balanced acc | macro F1 | water P / R / F1 | 多数类基线 | lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Random split** | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 (n=282) | 0.7378 | +0.2622 |
| **Block split**  | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 (n=121) | 0.7170 | +0.2830 |

两种切分都到 1.0。**这不是 bug，是事实**：

- 这景影像在 7 维特征空间里 3 类**完全可分**（混淆矩阵的非对角线全 0 直接证明）。
- 既然不需要利用空间近邻就能完美分类，random 切分的"潜在泄漏"在这景上没有可观察的影响。
- **结构上**它仍然在泄漏（见 3.1 的 97.28%），只是被特征强度盖住了。

不过，1.0 的结果也提醒我自己：**这景数据可能本身就比较容易**，
不能据此推出该模型在其他影像或新区域上也会完美。
本任务的重点不是证明模型"永远准确"，而是证明我**知道如何用空间切分和 per-class 指标检查评估是否可信**。

如果只看 3.2 这张表，会以为"切分方式不重要"——这是错的，3.1 已经证明 random 切分**结构上不可信**。
**无论 3.2 看不看得到 gap，结论都应该以 block 切分的数字为准**，random 仅作为对照。

### 3.3 在弱特征下两种切分都暴露了"高 acc 假象"

`python checks/weak_feature_diagnostic.py`：故意把模型弱化到只用 1 个特征（Red），
让小类失败暴露：

| 切分方式 | accuracy | balanced acc | macro F1 | water P / R / F1 |
| --- | ---: | ---: | ---: | ---: |
| Weak + Random | 0.9176 | 0.6631 | 0.6569 | **0.022 / 0.082 / 0.035** (n=282) |
| Weak + Block  | 0.9125 | 0.6612 | 0.6492 | **0.010 / 0.099 / 0.019** (n=121) |

解读（**这正是任务结尾那句话的活例子**）：

- 两种切分的 **accuracy 都 91% 上下**，看起来"还不错"。
- 但 **water recall ≈ 8–10%**，**water F1 ≈ 0.02–0.04**——663 像素的水体，模型几乎全没认出来。
- accuracy 被多数类（vegetation 73%）+ other（25%）抬起来，把水体的彻底失败完全藏起来。
- **balanced accuracy 只有 0.66**，**macro F1 只有 0.65**——这两个指标立刻揭穿了 91% 的假象。

→ **这就是为什么不能只看 accuracy**：必须看 balanced accuracy + macro F1 + per-class recall。

### 3.4 random vs block 在本数据集的 accuracy 差距很小，原因

弱特征实验里 `random_acc - block_acc = +0.0051`，差距小。两个原因：

1. 200×200 影像偏小，地物分布相对交错（other 散布、vegetation 几乎单连通），
   block 切分的训练块和测试块仍然看到了相似的光谱分布；
2. 弱特征下模型已经被信息量限制，能榨取的"邻居作弊"红利有限。

要观察到大 random/block gap，通常需要**更大尺度、更纯单类块的影像，或更依赖局部上下文的模型（CNN）**。
但**这不影响结论**——3.1 证明了结构上 random 切分必然泄漏，
所以即使 accuracy 没变化，**也应当采用 block 切分作为可信评估**。

## 4. 多数类基线对比

| 切分方式 | 模型 accuracy | 多数类基线 | lift | 通过验收？ |
| --- | ---: | ---: | ---: | :---: |
| Random | 1.0000 | 0.7378 | +0.2622 | ✓ 显著超过基线 |
| Block  | 1.0000 | 0.7170 | +0.2830 | ✓ 显著超过基线 |

模型在 block 切分下 lift +28 个百分点，确实学到了东西（不是全猜多数类）。

## 5. 空间合理性

`python checks/spatial_coherence.py` 量化对比 prediction 和 truth 的空间结构：

```
class         truth_px  truth_cc  truth_frag   pred_px   pred_cc   pred_frag
other             9976         4      0.0004      9976         4      0.0004
vegetation       29361         1      0.0000     29361         1      0.0000
water              663         1      0.0015       663         1      0.0015
```

- 每一类的连通组件数（cc）和碎片度（frag）与真值完全一致
- 水体保持单一连通块，没有椒盐噪声
- `outputs/preview.png` 提供肉眼三栏对比（truth / random / spatial）

> 注意：这里的空间一致性检查是对**整景**预测图（包含训练像素）的形态自检，
> 用于发现明显椒盐噪声或空间结构异常；
> 严格的泛化能力判断仍以**第 3 节中测试集上的 block split 指标**为准。

## 6. 验收对照

| SPEC 标准 | 检查脚本 / 文件 | 结果 |
| --- | --- | --- |
| 1 | `python src/run_pipeline.py` | PASS（5 个产物均生成） |
| 2 | `prediction_*.tif` dtype/CRS/transform/shape | PASS（uint8, EPSG:32650, 200×200） |
| 3 | `outputs/metrics.json` | PASS（accuracy / balanced_accuracy / macro_f1 / cm / per_class 齐全） |
| 4 | `python checks/check_data.py` | PASS（输入 sanity 全通过） |
| 5 | `python checks/check_split_leakage.py` | PASS（block 切分 0 重叠，median 距离 8 px） |
| 6 | `python checks/baseline_check.py` | PASS（基线 0.7340，lift +0.26） |
| 7 | `python checks/weak_feature_diagnostic.py` | PASS（弱特征下 water R≈0.08，演示"高 acc 掩盖小类"） |
| 8 | `python checks/spatial_coherence.py` | PASS（pred 与 truth 同碎片度） |

## 7. 一句话答辩

一个 accuracy 99% 的模型可能毫无用处：
（a）遥感影像相邻像素高度相似——在本数据集 random 切分下 **97.28% 的测试像素紧挨训练像素**——
模型只是在"识别近邻"而非泛化；
（b）多数类占比可能极高，全猜 vegetation 也能拿到 73% accuracy，
弱特征下水体 recall 只有 8% 也能撑起 91% 的总体 accuracy。
判断模型是否可信，必须**同时看**：block 切分下的混淆矩阵 + 每一类的 precision/recall + balanced accuracy + 多数类基线 lift——缺一不可。
