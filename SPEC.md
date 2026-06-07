# 任务规格：植被/水体二分类工具

## 目标

基于一景多波段卫星影像与同范围地面真值，训练一个像素级分类器，
把每个像素分到 `0=其他 / 1=植被 / 2=水体` 三类中，
并对模型给出**严格、不被准确率欺骗**的评估。

不是为了"准确率高"——是为了能**判断模型可不可信**。

## 输入

- `data/scene.tif`：4 波段 GeoTIFF
  - Band 1 = Red, Band 2 = NIR, Band 3 = Green, Band 4 = SWIR
  - 数据类型 float32，已经是反射率（无 SCALE/OFFSET）
  - shape = 200 × 200，CRS = EPSG:32650，30 m 像元
  - 无 nodata
- `data/labels.tif`：1 波段 uint8 GeoTIFF，与 scene.tif 同 CRS / transform / shape
  - 类别：`0 = 其他`，`1 = 植被`，`2 = 水体`
  - 类别占比（已统计）：
    - `0`：9 976 像素（24.94%）
    - `1`：29 361 像素（73.40%）
    - `2`：**663 像素（1.66%）**——极不平衡，是评估的重点

## 方法

- **模型**：`sklearn.ensemble.RandomForestClassifier`
  - 选择理由：小数据（4 万像素）下稳定、可解释、不需要调参；
    支持 `class_weight="balanced"` 直接处理类别不平衡。
- **特征**（每像素 7 维）：
  - 4 个原始波段：Red, NIR, Green, SWIR
  - NDVI  = (NIR − Red) / (NIR + Red)        ——植被指数
  - NDWI  = (Green − NIR) / (Green + NIR)    ——McFeeters 水体指数
  - MNDWI = (Green − SWIR) / (Green + SWIR)  ——Modified NDWI，对水体识别在含建成区/阴影时更稳
  - 加这三个指数是因为植被和水体在它们上的差异远比单波段大，是该类问题的经典做法。
- **超参数**：`n_estimators=200, min_samples_leaf=2,
  class_weight="balanced", random_state=42`。
  不在本任务里做超参搜索（非目标）。

## 训练 / 测试切分（关键）

像素之间**不独立**：相邻像素几乎一模一样。
所以本工具同时跑两种切分，把结果**都报出来**：

1. **Random split**：把 40 000 像素随机打乱，60% 训练 / 40% 测试。
   *会有数据泄漏的风险*——故意保留作为对照。
2. **Block split**：把 200×200 切成 32×32 的网格区块（共 49 块），
   按区块整体 60/40 划分。同一块内所有像素要么全进训练、要么全进测试，
   避免了"同一空间块内像素被随机拆开"的泄漏。

随机种子均为 42，可复现。

### 为什么 block 切分更可信（必答题）

- 遥感影像具有**空间自相关**——相邻像素的反射率高度相似，因为它们大概率属于同一地物。
- 随机按像素切分的后果：测试像素的 4-邻域里几乎一定有训练像素，
  模型不需要"理解光谱-地物关系"也能在测试集上接近完美——它只是在做"我在训练时见过和你长得一模一样的邻居"这件事。
- 这种 accuracy **不能外推到新影像 / 新区域**，因为模型并没有学到泛化所需的判别边界。
- block 切分把 32×32 = 1024 个像素整体分给训练或测试，
  保证同一个 block 不会同时出现在训练集和测试集中，
  **避免了"同一空间块内像素被随机拆开"的泄漏**。
  需要注意：block 边界处仍可能有相邻的训练/测试像素（一个 block 分到 train、紧挨它的 block 分到 test），
  因此 block 切分**不是完全消除空间自相关**，而是相比 random split 更诚实、更保守的评估方式。
  `checks/check_split_leakage.py` 用 block 交集和最近训练像素距离两个角度量化这一点。
- 因此评估模型的真实泛化能力，**应以 block 切分为主**，
  random 切分仅作为对照，用来暴露随机像素切分下的潜在空间近邻泄漏风险。

## 评估指标

任务硬性要求"不能只看 accuracy"。本工具在两种切分下都报告：

- 整体 **accuracy**
- **balanced accuracy**（每类 recall 的算术平均；类别不平衡下比 accuracy 更稳）
- **多数类基线 accuracy**（"全猜 vegetation"）和 lift over baseline
- **混淆矩阵**（3×3）
- **每一类的 precision / recall / F1 + support**
- **macro F1**（每类等权平均的 F1，对小类友好）

最终判断模型是否可信，看**水体类的 recall / F1**，而不是看 overall accuracy。

## 输出

所有产物写入 `outputs/`：

| 文件                              | 内容                                                              |
| --------------------------------- | ----------------------------------------------------------------- |
| `outputs/prediction_random.tif`   | random 切分模型在整景上的预测（uint8，nodata=255，同地理参考）     |
| `outputs/prediction_spatial.tif`  | block 切分模型在整景上的预测（uint8，nodata=255，同地理参考）      |
| `outputs/preview.png`             | 真值 / random / spatial 三栏对比图                                |
| `outputs/metrics.json`            | 类别分布 / 模型参数 / 两种切分的全部指标                           |
| `outputs/report.txt`              | 人读的纯文本评估报告                                               |

## 验收标准（必须可执行）

1. 运行 `python src/run_pipeline.py` 成功，`outputs/` 下出现 5 个文件。
2. `prediction_random.tif` 和 `prediction_spatial.tif` 的 dtype = uint8，CRS / transform / shape 与 `scene.tif` 完全一致。
3. `metrics.json` 中 `splits.random` 和 `splits.block` 两段都有
   accuracy / balanced_accuracy / macro_f1 / confusion_matrix /
   per_class（含 0/1/2 三类的 precision/recall/F1）。
4. `python checks/check_data.py` 输入数据所有断言通过。
5. `python checks/check_split_leakage.py` **block 切分中 train_blocks ∩ test_blocks = ∅**，
   且打印 random / block 两种切分下 test 像素到最近 train 像素的距离分布。
6. `python checks/baseline_check.py` 输出多数类基线 ≈ 73.4%，
   并打印两种切分相对基线的 lift。
7. `python checks/weak_feature_diagnostic.py` 用单一弱特征（仅 Red）
   重跑两种切分；输出明确显示 water recall 远低于 overall accuracy
   （演示"高 accuracy 掩盖小类失败"）。
8. `python checks/spatial_coherence.py` 比较 prediction 与 labels 中
   每一类的连通组件数与碎片度（4-邻域）。

## 非目标（明确不做）

- 不做超参搜索 / 模型选型对比（只用 RF + 一组合理默认）。
- 不做后处理（形态学闭合、CRF、滤波等去椒盐噪声）。
- 不做交叉验证；只用一次切分配上 majority baseline + 弱特征对照来诊断。
- 不做迁移到新影像（只服务于当前 scene.tif + labels.tif）。
- 不做交互式 GUI / Web UI，只提供 CLI。
- 不做时序分析。
