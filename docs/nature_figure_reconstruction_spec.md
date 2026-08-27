# KT 论文图件 Nature 风格重构规范（v31）

本规范仅针对基于公开聚合结果的知识追踪论文图件。它不适用于仓库中其他项目的 `nature-v30` 图件，也不读取原始学生记录、学生级预测、划分、逐技能参数或检查点。

## 设计语言

图件遵循紧凑、克制、数据优先的 Nature 风格：白色背景；无上、右边框；0.6 pt 深灰坐标轴；浅灰水平网格；Arial/Helvetica 优先的无衬线字体；色盲友好的深蓝 `#1B4F72`、青绿 `#2A7F7F`、赭橙 `#C65D3B` 与中性灰 `#6E7781`；直观面板标记 `a`/`b`；图例无边框；文本不依赖颜色单独传达含义。所有数值映射、轴范围、误差条和图例均由聚合 JSON 可复现生成。

所有图保留 `Figure` 写法。误差条严格按图注明确：单运行基线为 1,000 次 student-cluster bootstrap 95% interval；多种子图为 across-seed SD；三种子点/横线为观测点与均值，非推断区间。

## 命名与重构映射

| 论文位置 | 原聚合证据 | 重构文件前缀 | 视觉策略 |
| --- | --- | --- | --- |
| Figure 1 | 主分析基线聚合 | `figure-01-student-disjoint-baselines` | 三柱基线图；明确区分 bootstrap error bar 和 DKT 三种子开放点。 |
| Figure 2 | 200/500/full 窗口敏感性 | `figure-02-training-window-sensitivity` | 三点折线与 across-seed SD；统一 y 轴，不作等效性暗示。 |
| Figure 3 | 三种子主消融 | `figure-03-primary-ablation` | 三组 jitter 点、均值线与数值标签。 |
| Figure 4 | 标签翻转鲁棒性 | `figure-04-label-inversion-sensitivity` | 两条带标记曲线与 capped SD error bars。 |
| Figure S1 | 概率质量聚合 | `figure-s01-probability-quality` | Brier/ECE 分组柱 + reliability curve 双面板。 |
| Figure S2 | 20 epoch 探索性轨迹 | `figure-s02-exploratory-budget` | 三条验证轨迹、选中 checkpoint 和固定 8-epoch 参考线。 |
| Figure S3 | BKT 聚合诊断 | `figure-s03-bkt-diagnostics` | 参数 P05/P50/P95 区间 + 固定迭代稳定性对比双面板。 |

## 技术交付物

每个前缀输出同一数据映射下的 `.pdf`（首选矢量提交文件）、`.svg`（可编辑矢量源）、`.png`（600 dpi，用于 DOCX 嵌入）和 `.tif`（600 dpi，印刷交付）。`figure_manifest.json` 为每幅图记录聚合输入、SHA-256、轴/误差条解释、像素和 DPI。`

源文件 `scripts/reconstruct_kt_figures_nature.py` 及其生成的 `data/figure_data_kt_nature_v31.json` 均上传公开仓库；私密层保留一份相同逻辑的源码、主稿/补充信息插图与审计记录。
