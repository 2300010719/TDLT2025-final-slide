# Task 2: Loss-Curve Prediction Across Learning-Rate Schedules

## 1. 问题背景与目标

本项目关注不同 learning-rate schedule 下的 loss curve prediction 问题。给定一个学习率调度下已经观测到的训练损失曲线，我们希望预测另一个学习率调度下的训练损失变化趋势。

这一问题的动机是：在大模型训练中，完整训练一次模型成本很高。如果能够通过已有训练曲线预测不同学习率调度下的 loss curve，就可以帮助我们更快比较不同训练策略的效果，减少调参和重复训练成本。

本项目主要围绕以下问题展开：

- Tissue et al. 2024 和 Luo et al. 2024 的 loss-curve prediction 方法能否在给定数据上被复现？

- 在 cosine 学习率调度上拟合得到的模型，能否准确预测 WSD 学习率调度下的 loss curve？

- cosine、WSD 和 8-1-1 三种学习率调度之间，哪一种迁移方向的预测效果最好？

- 在复现已有方法的基础上，是否可以提出新的拟合方法，进一步提升跨调度预测性能？

## 2. 数据处理与实验设置

本实验使用课程提供的 gpt_loss+lrs.pkl 数据文件。该文件包含不同学习率调度下的训练记录，每条曲线包括训练 step、loss 和 learning rate。

实验中使用三种学习率调度：

- cosine：余弦退火学习率调度。

- WSD：warmup-stable-decay 调度，即先 warmup，再保持稳定学习率，最后 decay。

- 8-1-1：训练过程大致分为 80% 主训练阶段、10% 过渡阶段和 10% 衰减阶段。

数据处理流程如下：

- 从 pkl 文件中读取每种 schedule 对应的 loss 和 learning rate。

- 对每条曲线按 step 排序，并去除缺失值。

- 在拟合时从完整曲线中采样一部分训练点，以降低拟合计算量。

- 评估时跳过最早期训练不稳定阶段，从 step 500 之后计算误差。

- 使用相同的评估指标比较所有方法。

主要评估指标包括：

- MAE：平均绝对误差。

- RMSE：均方根误差。

- MAPE：平均相对误差，是本实验最主要的比较指标。

- R2：拟合优度。

- final relative error：最终 step 上的相对误差，用于衡量最终 loss 预测是否准确。

## 3. 复现实验方法

本项目首先复现两类已有方法：Tissue et al. 2024 和 Luo et al. 2024 的 MultiPowerLaw 方法。

### 3.1 Tissue2024 方法

Tissue2024 方法的核心思想是，loss curve 的变化不仅与训练步数有关，也与累计学习率以及学习率退火过程有关。

本实验中使用的形式可以写为：

```text
L(t) = L0 + A * S1(t)^(-alpha) - C * S2(t)
```

其中：

- S1(t) 表示累计学习率。

- S2(t) 表示由学习率下降产生的退火动量项。

- L0, A, alpha, C 是需要拟合的参数。

该方法的直觉是：累计学习率越大，模型训练越充分，loss 越低；同时学习率退火会带来额外的 loss drop。

### 3.2 LuoMPL 方法

Luo et al. 的 MultiPowerLaw 方法进一步对学习率变化引起的 loss drop 进行建模。其形式可以写为：

```text
L(t) = L0 + A * S1(t)^(-alpha) - B * LD(t)
```

其中：

- S1(t) 仍然是累计学习率。

- LD(t) 是由 learning-rate gap 和 multi-power law 公式得到的 loss-drop 项。

- L0, A, alpha, B 是拟合参数。

- 形状参数 C, beta, gamma 参考官方 MultiPowerLaw 仓库中的设置。

本实验尽量参考官方实现方式，在 cosine 曲线上拟合参数，然后迁移到 WSD 或其他调度上进行预测。

## 4. 主复现实验结果：cosine -> WSD

主任务要求是在 cosine 学习率调度上拟合模型，并在 WSD 学习率调度上评估预测性能。

实验结果如下：

| 方法 | 迁移方向 | MAPE | RMSE | Final Relative Error |
| --- | --- | --- | --- | --- |
| Tissue2024 | cosine -> WSD | 1.9611% | 0.06955 | 1.1268% |
| LuoMPL | cosine -> WSD | 1.9314% | 0.06875 | 0.1889% |

从结果可以看到，两个 baseline 方法都能够较好预测 WSD 曲线的整体趋势。LuoMPL 的 MAPE 略低于 Tissue2024，说明整体预测误差更小。同时，LuoMPL 的 final relative error 明显更低，说明它对训练末端 loss 的预测更准确。

因此，在本实验数据上，LuoMPL 是复现 baseline 中表现更好的方法。

## 5. 复现中的关键差异与困难

在复现过程中，主要困难来自以下几个方面。

第一，论文和官方代码中的模型形式并不总是可以直接套用到当前数据。不同数据集、模型规模和训练长度会导致 loss curve 的形状存在差异，因此需要对参数初始化、拟合点采样和数值稳定性进行处理。

第二，loss curve 在训练早期变化非常剧烈。如果直接使用所有 step 计算误差，早期不稳定区域会对指标产生较大影响。因此本实验从 step 500 之后开始评估，以更公平地比较中后期预测效果。

第三，跨调度预测比同调度拟合更困难。一个模型在 cosine 曲线上拟合得很好，并不一定能准确外推到 WSD。原因是不同学习率调度的 decay 位置、稳定阶段长度和 learning-rate gap 都不同，模型可能学习到源调度特有的曲线形状。

## 6. 迁移拓展实验：三种调度之间的预测

除了主任务中的 cosine -> WSD，本项目进一步测试了 cosine、WSD 和 8-1-1 三种调度之间的六种有向迁移。

平均 MAPE 结果如下：

| 迁移方向 | Mean MAPE |
| --- | --- |
| WSD -> 8-1-1 | 1.5004% |
| 8-1-1 -> WSD | 1.5324% |
| WSD -> cosine | 1.6287% |
| 8-1-1 -> cosine | 1.7615% |
| cosine -> WSD | 1.9462% |
| cosine -> 8-1-1 | 1.9763% |

结果显示，最佳平均迁移方向是：

```text
WSD -> 8-1-1
```

其平均 MAPE 为 1.5004%。

这说明 WSD 曲线作为源调度时，可能包含更丰富的训练阶段信息。WSD 具有 warmup、stable 和 decay 三个明显阶段，因此模型能从中学习到更完整的 loss 变化结构。相比之下，cosine 调度较平滑，作为源调度时提供的 schedule phase 信息较少，因此迁移到 WSD 和 8-1-1 时效果相对较弱。

## 7. 已有方法的局限性

通过复现实验和迁移拓展实验，可以观察到已有方法存在一些局限性。

第一，Tissue2024 和 LuoMPL 都依赖人工设计的 schedule 特征，例如累计学习率、学习率差分和 loss-drop 项。这些特征在某些调度之间迁移效果较好，但不一定适合所有调度。

第二，已有方法更关注 learning-rate schedule 对 loss 的影响，但对 loss curve 本身的形状建模仍然有限。例如，实际 loss curve 可能同时包含长期 power-law 下降和早期快速 exponential-like 下降，仅使用单一 power law 可能不够灵活。

第三，源调度拟合效果好不代表迁移效果好。一些方法可以很好拟合 cosine 曲线，但迁移到 WSD 后误差反而较大。这说明跨调度预测更需要关注模型的泛化结构，而不是只追求源曲线拟合误差。

第四，残差修正方法容易过拟合源调度的局部噪声。实验中 StepResidual 和 EffectiveResidual 在源曲线上拟合效果不错，但迁移到 WSD 后表现较差，说明简单残差修正并不一定能提升跨调度预测。

## 8. 方法发展：提出的新方法

在复现已有方法后，本项目进一步提出并比较了多种新的 loss curve fitting 方法。核心思想是：除了 schedule-aware correction，还需要更灵活地建模 loss curve 自身的形状。

本项目尝试的方法包括：

| 方法 | 说明 |
| --- | --- |
| Dev1-EffectiveTime | 使用累计有效训练时间替代原始 step |
| Dev2-StepResidual | 在 step-based power law 上加入残差修正 |
| Dev3-EffectiveResidual | 在 effective-time power law 上加入残差修正 |
| Dev4-VelocityMatched | 加入 loss velocity matching，约束 loss 下降速度 |
| Dev5-HybridTissueMPL | 结合 Tissue 和 MultiPowerLaw 的 correction |
| Dev6-CurvatureMatched | 同时加入速度和曲率约束 |
| Dev7-PowerExponential | 使用 power law + exponential decay 混合形式 |
| Dev8-TimeEnsemble | 使用多个时间坐标下的 ensemble |
| Dev9-TissueVelocity | 在 Tissue2024 上加入 velocity matching |
| Dev10-LuoMPLVelocity | 在 LuoMPL 上加入 velocity matching |
| Dev11-PowerExpVelocity | 在 PowerExponential 上加入 velocity matching |

其中，最终表现最好的方法是：

```text
Dev11-PowerExpVelocity
```

## 方法发展思路

首先，我们设计并测试了方法 1 到方法 7。这一阶段的目标是比较不同建模思想对 cosine -> WSD 跨调度预测的影响。

这些方法大致可以分为几类：

- 时间坐标改造类方法代表方法是方法1 EffectiveTime。它不直接使用原始 step，而是根据 learning rate 构造累计有效训练时间。这样可以让不同学习率调度下的训练进度具有更可比的时间尺度。

- 残差修正类方法包括方法2 StepResidual 和方法3 EffectiveResidual。它们先拟合一个基础 power-law 曲线，再对剩余误差进行修正。设计动机是希望捕捉基础模型无法解释的局部结构。

- 速度与曲率约束类方法包括方法4 VelocityMatched 和方法6 CurvatureMatched。普通拟合只要求预测 loss 数值接近真实 loss，但这些方法进一步要求预测曲线的下降速度，甚至二阶曲率，也与真实曲线接近。这样可以避免模型只在点值上拟合较好，却在趋势上外推不稳定。

- 混合模型类方法包括方法5 HybridTissueMPL 和方法7 PowerExponential。方法5 尝试结合 Tissue2024 和 LuoMPL 的 schedule correction；方法7 则将 power-law 长期趋势和 exponential decay 快速下降项结合起来，希望更灵活地刻画完整 loss curve。

通过比较方法1到方法7的结果，我们得到两个重要发现。

第一，方法4 VelocityMatched 相比方法1 EffectiveTime 有提升。这说明在 loss curve prediction 中，只拟合 loss 的数值是不够的。加入 loss 下降速度约束后，模型能够更好保持曲线趋势，从而提升跨调度预测效果。

也就是说，velocity matching 是一个有效的改进方向。

第二，方法7 PowerExponential 在前七个方法中表现最好。这说明当前数据中的 loss curve 可能并不是单一 power law，而是同时包含两种结构：

```text
长期训练趋势：power-law decay
```

```text
早期/中期快速下降：exponential-like decay
```

因此，方法7 使用

```text
L(t) = L0 + A * t^(-alpha) + E * exp(-lambda * t / n)
```

比单纯的 power law 或 schedule correction 更能刻画当前数据的曲线形状。

### 将有效思想迁移到基础模型上

```text
Velocity matching 可以提升模型效果；
```

```text
PowerExponential 是当前最强的曲线主模型。
```

因此，我们进一步将 velocity matching 这个思想直接加入不同主模型中，观察它是否具有通用性。

具体来说，我们构造了以下方法：

| 方法 | 思路 |
| --- | --- |
| Dev9-TissueVelocity | 在 Tissue2024 主模型上加入 velocity matching |
| Dev10-LuoMPLVelocity | 在 LuoMPL 主模型上加入 velocity matching |
| Dev11-PowerExpVelocity | 在 PowerExponential 主模型上加入 velocity matching |

在这些方法中，velocity matching 的权重统一固定为：

```text
velocity_weight = 0.50
```

### 第二阶段的发现

实验结果显示，velocity matching 在不同主模型上的效果并不完全相同。

对于 Tissue2024 和 LuoMPL，加入 velocity matching 后只有小幅改善。这说明 velocity matching 本身有帮助，但它的效果受到主模型表达能力限制。如果主模型本身对 loss curve 形状刻画不足，仅加入速度约束也难以带来大幅提升。

而对于 PowerExponential，加入 velocity matching 后得到了最好的结果，即：

```text
Dev11-PowerExpVelocity
```

这说明：

```text
强主模型结构 + 趋势约束
```

比单独改造 schedule 特征或单独加入残差修正更有效。

### 方法发展总结

整体来看，我们的方法发展不是一次性提出最终模型，而是一个逐步筛选和组合的过程。

首先，我们测试了方法1到方法7，发现：

```text
方法4说明 velocity matching 是有效方向；
```

```text
方法7说明 PowerExponential 是更强的曲线主模型。
```

然后，我们将 velocity matching 进一步应用到不同基础模型上，包括 Tissue2024、LuoMPL 和 PowerExponential。

最终结果表明，最佳方法是：

```text
Dev11-PowerExpVelocity
```

它结合了两点优势：

- PowerExponential 提供更灵活的 loss curve 形状建模能力。

- Velocity matching 约束预测曲线的下降趋势，使外推更加稳定。

因此，我们最终提出的方法可以理解为：

```text
Power-law 长期趋势
```

```text
+ Exponential 快速下降项
```

```text
+ Loss velocity matching 趋势约束
```

这个组合在 cosine -> WSD 预测任务上取得了最好的效果。

## 9. 核心方法：PowerExponential

Dev7-PowerExponential 的形式为：

```text
L(t) = L0 + A * t^(-alpha) + E * exp(-lambda * t / n)
```

这个模型由两部分组成：

- A * t^(-alpha)：power-law 项，用于描述长期训练中的 scaling trend。

- E * exp(-lambda * t / n)：exponential 项，用于描述训练早期和中期的快速下降。

设计这个方法的原因是，实际 loss curve 往往并不是单一 power law。特别是在训练早期，loss 会快速下降；而在训练后期，下降速度逐渐放缓，更接近 power-law 行为。因此，power law 和 exponential decay 的组合可以更灵活地刻画完整 loss curve。

## 10. 核心方法：Velocity Matching

在 Dev11-PowerExpVelocity 中，本项目进一步加入 velocity matching。这里的 velocity 指的是 loss 在相邻 step 之间的下降速度。

普通拟合只要求预测的 loss 数值接近真实 loss，而 velocity matching 进一步要求预测曲线的下降趋势也接近真实曲线。

直观来说，如果两个模型在某些点上的 loss 值相近，但一个模型下降过快、另一个模型下降过慢，那么它们在未来外推时可能表现不同。Velocity matching 希望让模型不仅拟合当前点，也拟合曲线变化的方向和速度。

本实验中，所有 velocity matching 方法统一固定：

```text
velocity_weight = 0.50
```

这样做的原因是避免使用目标曲线结果调参，使方法比较更加清晰。

## 11. 方法发展结果对比

在主任务 cosine -> WSD 上，不同方法的结果如下：

| 方法 | MAPE | RMSE | Final Relative Error |
| --- | --- | --- | --- |
| Tissue2024 baseline | 1.9611% | 0.06955 | 1.1268% |
| LuoMPL baseline | 1.9314% | 0.06875 | 0.1889% |
| Dev1-EffectiveTime | 1.8612% | 0.06695 | 1.5954% |
| Dev4-VelocityMatched | 1.8587% | 0.06695 | 1.5999% |
| Dev7-PowerExponential | 1.5900% | 0.05654 | 1.2032% |
| Dev11-PowerExpVelocity | 1.5887% | 0.05651 | 1.1990% |

最佳方法是 Dev11-PowerExpVelocity，其 MAPE 为：

```text
1.5887%
```

相比最佳 baseline LuoMPL 的 1.9314%，MAPE 从 1.9314% 降低到 1.5887%，相对提升约：

```text
17.7%
```

这说明 PowerExponential 结构对当前 loss curve 更合适，而 velocity matching 在此基础上带来了小幅进一步提升。

## 12. 结果分析

从实验结果可以得到几个观察。

第一，PowerExponential 系列方法明显优于 Tissue2024 和 LuoMPL baseline。这说明当前数据中的 loss curve 可能同时包含 power-law 长期趋势和 exponential-like 快速下降阶段。单独依赖累计学习率或 loss-drop correction 可能不足以完全描述曲线形状。

第二，Dev11 相比 Dev7 的提升较小，但仍然是最优方法。这说明 velocity matching 对 PowerExponential 有一定帮助，但主要提升仍来自 PowerExponential 的模型结构本身。

第三，Dev4-VelocityMatched 相比 Dev1-EffectiveTime 有轻微提升，说明速度约束可以改善迁移，但提升幅度有限。这可能是因为 velocity matching 只约束局部下降趋势，并不能完全解决不同调度之间的全局形状差异。

第四，Dev2 和 Dev3 这类 residual correction 方法表现较差。它们在源曲线中可能拟合到了局部残差和噪声，但这些残差不具备跨调度泛化能力。因此，简单加入残差项并不一定能提升迁移预测。

第五，迁移方向实验表明 WSD 是较好的源调度。WSD 的 warmup、stable 和 decay 阶段更清晰，可能提供了比 cosine 更丰富的 schedule 信息，因此 WSD -> 8-1-1 和 WSD -> cosine 的效果都较好。

## 13. 进一步思考

本实验说明，跨调度 loss curve prediction 的关键并不只是拟合源调度曲线，而是要学习能够迁移到其他调度的结构。

未来可以从以下方向进一步改进：

- 在更多模型规模和更多训练数据上验证 PowerExponential 是否稳定有效。

- 将 velocity_weight 放到独立验证任务上选择，而不是手动固定。

- 对 WSD 和 8-1-1 的 phase transition 进行更细粒度建模，例如分别建模 stable 阶段和 decay 阶段。

- 引入不确定性估计，不只给出单条预测曲线，也给出预测区间。

- 将 schedule-aware 特征与更灵活的 curve-shape model 结合，构造更强的跨调度预测模型。

## 14. 总结

本项目完成了 Task 2 要求的主要内容。

首先，我们复现了 Tissue2024 和 LuoMPL 两个已有 loss-curve prediction 方法，并在主任务 cosine -> WSD 上进行了评估。结果显示 LuoMPL 略优于 Tissue2024，是复现 baseline 中表现较好的方法。

其次，我们进行了三种学习率调度之间的迁移拓展实验，发现 WSD -> 8-1-1 是最佳迁移方向，说明 WSD 曲线可能包含更丰富的训练阶段信息。

最后，我们提出并比较了多种新的拟合策略。其中 Dev11-PowerExpVelocity 取得最佳结果，在 cosine -> WSD 上达到 1.5887% 的 MAPE，相比最佳 baseline LuoMPL 有明显提升。

整体来看，本实验表明：有效的跨调度 loss prediction 需要同时考虑 learning-rate schedule 的影响和 loss curve 自身的形状结构。
