# 文献与实验结果登记

## 文献来源与协议边界

| 编号 | 来源 | 可引用事实 | ASSISTments2009 AUC（原文） | 与本次实验的关系 |
| --- | --- | --- | --- | --- |
| L1 | Liu et al., pyKT, NeurIPS Datasets and Benchmarks 2022 | 指出自定义预处理和错误评估设置会限制可比性，并报告错误设置可能导致标签泄漏。 | 不适用 | 为本次学生级划分、公开脚本和协议披露提供依据。 |
| L2 | Liu et al., simpleKT, ICLR 2023 | 在 pyKT 的标准化实验中报告多方法比较；论文同时指出文献中 ASSISTments2009 的 DKT/AKT 报告值存在较大范围，受预处理和超参数影响。 | IEKT 0.7861±0.0027；AKT 0.7853±0.0017；LPKT 0.7814±0.0022；simpleKT 0.7744±0.0018；DKT+ 0.7547±0.0017。 | 可作为**标准化历史对照**，但不得视为本次 CSV 编码、80/10/10 学生级 split 的直接可比SOTA。 |
| L3 | Cheng et al., UKT, AAAI 2025 | 使用不确定性表征和Wasserstein self-attention；作者以80%学生序列训练、剩余20%测试，训练数据内进行5-fold CV，并搜索超参数。 | UKT 0.8563±0.0014；AKT 0.8474±0.0017；simpleKT 0.8413±0.0018；DKT 0.8226±0.0011；SAKT 0.7746±0.0017。 | 属于另一套处理/评估协议，数值只能并列展示并注明不可直接计算缺口。 |
| L4 | Huang et al., sparseKT, SIGIR 2023 | 提出k-sparse attention以缓解注意力KT在小规模教育数据上的过拟合风险，并提供开源实现入口。 | 该来源不在当前登记中提供可直接使用的AS2009统一数值。 | 支撑将注意力稀疏化作为后续模型扩展的候选，不用于本次GRU结果的性能断言。 |
| L5 | ASSISTmentsData 官方数据页及Terms of Use | 官方页提供 corrected collapsed CSV，提醒旧版含重复记录；使用条款禁止将匿名化学生数据交给他人。 | 不适用 | 决定原始数据及学生级派生数据不可上传至GitHub，私密目录仅本地受控保存。 |

## 本次已运行实验（本地受控数据）

所有下列实验使用官方 corrected collapsed CSV（SHA-256：`162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`）。预处理将未空缺 `skill_id` 视为**未分解的类别标签**，仅保留有至少2次交互的学生序列；最终有4,027名学生，按固定种子20260822划分为3,221 train / 403 validation / 403 test。测试中，prior与BKT得分了24,709行，DKT得分了24,306个可预测 next interactions。所有 bootstrap 区间均按学生聚类重采样1,000次。

| 轮次 | 方法 | 主要配置 | 测试ROC-AUC | 不确定性/重复 | 解释边界 |
| --- | --- | --- | ---: | --- | --- |
| R1/R2 | Skill-prior | train student的Laplace平滑每技能均值 | 0.6230 | 95% CI [0.6102, 0.6356] | 参照项，不使用历史上下文。 |
| R1/R2 | per-skill BKT EM | 两状态BKT；训练学生拟合 | 0.7236 | 95% CI [0.7068, 0.7412] | 教学性实现；非完整的文献复现。 |
| R2 | DKT-64-Adam | embedding/hidden 64；lr 0.002；无dropout/weight decay；8 epoch；validation选epoch | 0.7654 | 3 seeds；SD 0.00112 | 本次实验的清洁GRU基线。 |
| R2 | DKT-64-AdamW | 与DKT-64-Adam相同，仅AdamW `weight_decay=1e-4` | 0.7654 | 3 seeds；SD 0.00112 | 相对清洁Adam均值变化约−0.00000006，未显示实际改善。 |
| R2 | DKT-96-AdamW | embedding/hidden 96；dropout 0.1；lr 0.001；`weight_decay=1e-4`；8 epoch | 0.7657 | 3 seeds；SD 0.00088 | 相对清洁DKT-64-Adam均值约+0.00030，小于基线跨种子SD；不能作为性能提升主张。 |
| R3 | DKT-64-Adam，训练端10%随机标签翻转 | 与清洁配置一致，测试/验证不扰动 | 0.7560 | 3 seeds；SD 0.00128 | 相对清洁同配置均值约−0.00942；仅为合成训练标签扰动测试。 |
| R3 | DKT-64-AdamW，训练端10%随机标签翻转 | 同上，`weight_decay=1e-4` | 0.7560 | 3 seeds；SD 0.00128 | 相对受扰Adam均值约+0.00000014；未见权重衰减的鲁棒性改善。 |

## 可直接用于论文的谨慎结论

在本次固定的CSV编码、学生级80/10/10 split和八个训练epoch下，64维GRU-DKT的多种子平均ROC-AUC为0.7654。添加`1e-4`解耦权重衰减并未给出可观察的均值增益；增加容量、dropout和改用AdamW的组合候选仅带来约0.00030的均值差异，低于清洁基线的跨种子标准差。训练侧10%独立随机标签翻转使两种配置均下降约0.00942 AUC。该结果不支持将当前轻量优化称为性能提升或鲁棒性增强。

原稿记录的 student-disjoint DKT AUC为0.764（bootstrap 95% CI [0.758, 0.770]），与本次同为学生级实验的单一数值接近，但二者的数据处理、划分比例、训练实现和资格行数不同；它们不应合并为同一估计，也不能作为逐点复现声明。

## 永久链接

[L1] Liu, Z., et al. (2022). *pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models*. NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2206.11460  
[L2] Liu, Z., et al. (2023). *simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing*. ICLR. https://arxiv.org/html/2302.06881v2  
[L3] Cheng, W., et al. (2025). *Uncertainty-aware Knowledge Tracing*. AAAI-25. https://ojs.aaai.org/index.php/AAAI/article/view/35007/37162  
[L4] Huang, S., et al. (2023). *Towards Robust Knowledge Tracing Models via k-Sparse Attention*. SIGIR. https://doi.org/10.1145/3539618.3592073  
[L5] ASSISTmentsData. *Skill-builder data 2009–2010*. https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010  
[L6] ASSISTmentsData. *Terms Of Use For Using Data*. https://sites.google.com/site/assistmentsdata/termsofuseforusingdata
