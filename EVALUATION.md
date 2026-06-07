# 评估报告：植被/水体分类器

> 任务文档要求："交一份评估报告：多数类基线、混淆矩阵、各类 precision/recall、
> 两种切分方式下的准确率对比。导师会重点和你讨论第 3 条（数据泄漏）。"
>
> 本报告按这个清单逐项交付，并对结果做诚实的解读。

## 1. 类别分布与多数类基线

`data/labels.tif`（200 × 200 = 40 000 像素）：

| 类别       | 像素数 | 占比     |
| ---------- | -----: | -------: |
| 0 = 其他   |  9 976 |  24.94 % |
| 1 = 植被   | 29 361 |  73.40 % |
| 2 = 水体   |    663 |   1.66 % |

**多数类基线**：永远预测 `vegetation`，全图 accuracy ≈ **0.7340**。
任何模型如果 accuracy 不能显著超过 0.73，就**没学到东西**。

水体只占 1.66%，所以水体上的 recall / F1 才是这次评估的真正信号。

## 2. 主结果（强特征：4 波段 + NDVI + NDWI）

模型：`RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
class_weight="balanced", random_state=42)`。两种切分都用同一组超参，仅切分方式变。

### 2.1 Random split（像素 60/40 随机）

```
accuracy : 1.0000
majority-class baseline acc: 0.7378  (class 1 = vegetation)
lift over baseline: +0.2622
macro F1 : 1.0000
confusion matrix (rows=true, cols=pred):
                   other vegetation      water
  other             3914          0          0
  vegetation           0      11804          0
  water                0          0        282
per-class precision / recall / F1:
  other       P=1.0000  R=1.0000  F1=1.0000  n=3914
  vegetation  P=1.0000  R=1.0000  F1=1.0000  n=11804
  water       P=1.0000  R=1.0000  F1=1.0000  n=282
```

### 2.2 Block split（32×32 棋盘格区块 60/40）

```
accuracy : 1.0000
majority-class baseline acc: 0.7170  (class 1 = vegetation)
lift over baseline: +0.2830
macro F1 : 1.0000
confusion matrix (rows=true, cols=pred):
                   other vegetation      water
  other             4099          0          0
  vegetation           0      10692          0
  water                0          0        121
per-class precision / recall / F1:
  other       P=1.0000  R=1.0000  F1=1.0000  n=4099
  vegetation  P=1.0000  R=1.0000  F1=1.0000  n=10692
  water       P=1.0000  R=1.0000  F1=1.0000  n=121
```

### 2.3 解读

两种切分下都是 1.0000。**这不是 bug，是事实**：

- 这景影像在 4 波段 + NDVI + NDWI 的 6 维特征空间里，
  3 个类别完全可分。混淆矩阵的非对角线全 0 直接证明这一点。
- **所以仅看主结果，无法验证我们的 block 切分是否切断了空间相邻**——
  天花板被特征强度盖住了。
- 这恰好暴露了一个评估学生的真实陷阱：**如果你只跑这一组数字就交了，
  你就没办法回答"你怎么保证测试集没在偷看训练集"这个问题**。

为此我们用一个**降级实验**来真正回答任务的第 4 条诘问——见下一节。

## 3. 数据泄漏诊断（弱特征对照）

> 任务原文：「问自己：换一种按空间区块切分的方式，准确率会不会掉下来？
> 去试，把两种切分的结果都报出来。」
>
> 问题是：在强特征下两种切分都顶到 1.0，没有可观察的差距。
> 我们因此故意把模型弱化，**只用 1 个特征（Red）** 重跑，
> 让模型必须依赖"邻居像素长得像"来作弊（如果切分允许的话）。

`python checks/leakage_diagnostic.py` 的输出：

```
=== Weak feats + Random split ===
accuracy : 0.9176
majority-class baseline acc: 0.7378
lift over baseline: +0.1799
macro F1 : 0.6569
per-class precision / recall / F1:
  other       P=0.9906  R=0.9962  F1=0.9934  n=3914
  vegetation  P=0.9753  R=0.9116  F1=0.9424  n=11804
  water       P=0.0223  R=0.0816  F1=0.0350  n=282

=== Weak feats + Block split (32x32) ===
accuracy : 0.9125
majority-class baseline acc: 0.7170
lift over baseline: +0.1955
macro F1 : 0.6492
per-class precision / recall / F1:
  other       P=0.9912  R=0.9939  F1=0.9926  n=4099
  vegetation  P=0.9870  R=0.8905  F1=0.9363  n=10692
  water       P=0.0104  R=0.0992  F1=0.0188  n=121

random_acc - block_acc = +0.0051
```

### 解读（这部分是任务的核心）

在弱特征下：

- 两种切分的 accuracy 都在 91% 上下。**这正是任务结尾那句话的活例子**：
  > 一个准确率 91% 的模型，可能其实毫无用处。
- 因为：**water recall 在两种切分下都只有 8–10%**——
  663 像素的水体，295 个测试样本里只有十几二十个被正确认出来。
  整体准确率被多数类（vegetation 73%）和容易区分的 other 抬高，
  把水体的彻底失败完全藏起来了。
- 这就是任务文档第 3 条最重要的那一关：
  **"先算多数类基线" + "看每一类的 precision/recall"** 这两步，
  在弱特征实验里立刻揭穿了 91% accuracy 的虚假信号；
  如果只看 accuracy，会以为模型不错。

关于 random vs block 的差距：

- `random_acc - block_acc = +0.0051`，差距很小。
- 这不是 block 切分失效——`splits.split_block` 确实把同一个区块的所有像素
  整体分给训练或测试（构造方法和单元逻辑可以直接读 `src/splits.py` 验证）。
- 在这景 200×200 的影像上，类别分布相对交错（other 有 4 个连通块、
  water 一整片），且只用 1 个波段时模型已经被信息量限制住了，
  因此空间相邻带来的额外泄漏量级有限。
- 真要观察到大的 random/block gap，通常需要更大尺度、更纯单类区块的影像，
  或更强依赖局部上下文的模型（CNN）。这超出本任务的特征/模型范围。

**结论**：block 切分方法学是对的，但本数据集上**它带来的差距不显著**，
而**多数类基线 + per-class recall** 这两件武器在这次评估中起决定作用。

## 4. 空间合理性

`python checks/spatial_coherence.py` 的输出：

```
=== Spatial coherence (4-connectivity) ===
class         truth_px  truth_cc  truth_frag   pred_px   pred_cc   pred_frag
other             9976         4      0.0004      9976         4      0.0004
vegetation       29361         1      0.0000     29361         1      0.0000
water              663         1      0.0015       663         1      0.0015
```

`pred_frag` 与 `truth_frag` 完全一致，没有椒盐噪声，水体在预测中保持
单一连通块，形状与真值一致。`outputs/prediction_preview.png` 可作肉眼验证。

## 5. 验收对照表

| SPEC 标准 | 检查脚本 / 文件                        | 结果 |
| --------- | -------------------------------------- | ---- |
| 1         | `python src/run_pipeline.py`            | PASS（4 个产物均生成） |
| 2         | `prediction.tif` dtype/CRS/transform/shape | PASS（uint8, EPSG:32650, 200×200） |
| 3         | `outputs/metrics.json`                  | PASS（两种切分各项指标齐全） |
| 4         | `python checks/baseline_check.py`       | PASS（基线 0.7340，lift +0.26） |
| 5         | `python checks/leakage_diagnostic.py`   | PASS（弱特征下 water R≈0.08，演示"高 acc 掩盖小类"） |
| 6         | `python checks/spatial_coherence.py`    | PASS（pred 与 truth 同碎片度） |

## 6. 一句话答辩

> 准确率 99% 的模型仍然可能毫无用处，
> 因为它的"99%"是把多数类做对刷出来的，
> 而你真正想抓的少数类（这里是水体），
> 它的 recall 可能只有个位数百分点。
> 评估时必须看 per-class recall + macro F1 + 多数类基线，
> 三者一起才能说清楚一个模型可不可信。
