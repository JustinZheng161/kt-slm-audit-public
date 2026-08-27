# 数据来源与隔离登记

## ASSISTments2009 Skill Builder（主实验数据）

| 字段 | 已核验信息 |
| --- | --- |
| 官方页面 | https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010 |
| 访问日期 | 2026-08-27（GMT+8） |
| 数据范围 | 2009–2010 学年 ASSISTments Skill Builder 数学练习交互数据 |
| 修正说明 | 官方页面说明原链接所指数据存在重复记录，并提供一个 corrected version。 |
| 本次指定文件 | `skill_builder_data_corrected_collapsed.csv`（官方 Google Drive 文件 ID：`1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE`） |
| 行级语义 | 官方页面说明修正文件为每个 student–problem 一行；多技能会以 `skill1_skill2` 合并在同一行。 |
| 引用要求 | 官方页面要求在出版物中给出该精确数据页 URL，并引用 Feng, Heffernan, & Koedinger (2009)。 |
| 仓库隔离策略 | 原始 CSV、派生的学生 ID 划分和含原始交互的中间文件只进入私密数据仓库；开源代码仓库仅保留下载说明、校验和、模式文档、生成脚本和不可逆聚合结果。 |

> 该官方页面还说明：使用 Hint 或“Break this Problem Into Steps”的问题会被标记为 incorrect。这一编码规则将在预处理与论文数据部分保留为数据语义，而不将其重解释为未经验证的学习能力标签。

## 当前下载状态

| 工件 | 本地位置 | 状态 | 后续核查 |
| --- | --- | --- | --- |
| 官方修正原始 CSV | `private_data/raw/skill_builder_data_corrected_collapsed.csv` | 已下载 | SHA-256、列名、行数、学生/题目/技能计数 |
| 预处理数据 | `private_data/processed/`（待生成） | 未开始 | 预处理脚本版本、不可逆变化记录 |
| 学生级划分 | `private_data/splits/`（待生成） | 未开始 | 同一学生不跨 train/test 的集合交集检验 |

## 使用条款与 GitHub 同步限制

官方 Terms of Use 页面说明数据虽为匿名化数据，但仍受学生级数据与 FERPA 保护；使用者不得尝试识别个人、不得将数据交给他人，并须在发现可识别信息时删除并通知 ASSISTments。该页面还要求在发表中致谢 ASSISTments，并说明算法应公开 [D3]。因此，**不论仓库可见性为 public 或 private，原始 CSV、逐学生划分文件、逐学生预测、序列样本和其他可逆派生数据均不得同步至 GitHub**。私密数据目录在本地受控保存；开源代码仓库只能包含下载说明、官方链接、校验和、schema、预处理程序和不可逆聚合统计。

| 拟同步工件 | 是否可进入 GitHub | 处理方式 |
| --- | --- | --- |
| 原始 `skill_builder_data_corrected_collapsed.csv` | 否 | 仅保留在本地受控私密目录；仓库记录官方下载链接与 SHA-256。 |
| 按学生划分的 `train/test` ID 文件 | 否 | 仅本地受控保存；公开仓库提供确定性生成脚本而不上传 ID 列表。 |
| 逐学生预测、样本序列、调试日志 | 否 | 本地受控保存或清除；只公开不可逆聚合指标。 |
| 预处理、训练、评分与绘图代码 | 是 | 在移除数据路径、令牌和本地标识符后公开。 |
| 聚合指标、置信区间和图表 | 是 | 仅当不能回推个人记录，且配有生成脚本、配置与来源说明时公开。 |
| 校验和、schema、数据卡与官方链接 | 是 | 随开源代码仓库同步，用于复核输入版本。 |

## 引用

[D1] ASSISTmentsData. *Skill-builder data 2009–2010*. https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010  
[D2] Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). *Addressing the assessment challenge with an online system that tutors as it assesses*. User Modeling and User-Adapted Interaction, 19(3), 243–266. https://doi.org/10.1007/s11257-009-9063-7  
[D3] ASSISTmentsData. *Terms Of Use For Using Data*. https://sites.google.com/site/assistmentsdata/termsofuseforusingdata

## 扩展数据集入口（尚未下载）

官方旧版 ASSISTments2015 页面已将访问入口重定向至更新的数据页。作为第二个实验基准，它将沿用 ASSISTments 的官方使用条件和数据隔离要求，且只在新的数据页、下载对象、模式和许可均被再次核验后才会被纳入实验；目前不将搜索摘要中的样本数写入论文结果。

| 候选数据集 | 官方入口 | 用途 | 当前状态 |
| --- | --- | --- | --- |
| ASSISTments2015 Skill Builder | https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data | 检验跨时间 ASSISTments 迁移性 | 入口已核验；尚未下载/处理 |

### ASSISTments2015 已验证下载

官方更新页说明该数据为2015年采集的 ASSISTments 数据，只含学生响应量最高的100个 skill builders。已下载文件 `assistments2015_skill_builder.csv` 的本地受控聚合画像为：708,631行、19,917名学生、100个 `sequence_id`、正确率70.61%，SHA-256为`75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0`。该文件只含`user_id`、`log_id`、`sequence_id`和`correct`四列，不能假定其提供了与2009 corrected collapsed CSV相同的题目/技能语义；因此当前仅作为推荐/扩展基准，未混入本文的主实验结果。

### EdNet 已核验推荐信息

EdNet 官方仓库说明其提供自2017年开始、覆盖两年以上Santa学习平台的131,441,538次交互和784,309名学生；另包含13,169个题目、1,021个讲座和293类技能。该资源提供KT1至KT4四个层级的数据，层级越高包含的行为类型越多，并声明采用CC BY-NC 4.0许可。由于数据规模和行为丰富性，它适合作为后续可扩展性与多行为KT检验的权威候选，但不在本次受限算力的主实验中下载或计算。

| 推荐优先级 | 数据集 | 已核验规模 | 采用理由 | 数据/代码隔离要点 |
| --- | --- | --- | --- | --- |
| 1 | ASSISTments2015 | 708,631 CSV行；19,917学生；100个sequence ID（本地聚合画像） | 与主实验同为ASSISTments，利于测试跨年份迁移，且文件体量可控。 | 仅本地受控原始数据；开源仓库只放下载说明、SHA和脚本。 |
| 2 | EdNet（建议从KT1起） | 131,441,538交互；784,309学生；13,169题目；293技能（官方仓库） | 足够大且有KT1–KT4层级，可检验训练扩展性和多行为信息是否有益。 | 遵守CC BY-NC 4.0；下载的数据仍不应随代码仓库镜像，保留官方来源和版本信息。 |

### 第三个标准基准：Algebra2005

PSLC DataShop的KDD Cup 2010官方入口将任务定义为：根据智能辅导系统交互日志预测学生在数学问题上的表现，并提供开发/挑战数据划分、下载和引用格式 [D6]。在simpleKT的预处理统计表中，Algebra2005原始数据为809,694次交互、574个原始序列、210,710题、112个KC；其框架处理后为884,098次交互、4,712序列、173,113题和112个KC [D7]。原始与预处理数据的`sequences`数增加、题目数减少，反映了处理管线的重构作用；因此论文表格必须逐列标明采用的处理版本，不能将二者混写。

| 主实验/对照数据集 | 原始数据规模（可核验） | 标准化处理后规模（simpleKT） | 当前论文用途 |
| --- | --- | --- | --- |
| ASSISTments2009 corrected collapsed | 346,860行；4,217学生；26,688 problem ID；150个raw collapsed skill label（本地画像） | 337,415 interactions；4,661 sequences；17,737问题；123 KC | 本次实际运行的主实验；本地处理结果独立报告。 |
| ASSISTments2015 Skill Builder | 708,631行；19,917学生；100个sequence ID（本地画像） | 682,789 interactions；19,292 sequences；100 KC | 已收集，作为跨年份扩展；尚未纳入本文结果表。 |
| Algebra2005 | 809,694 interactions；574 sequences；210,710题；112 KC | 884,098 interactions；4,712 sequences；173,113题；112 KC | 标准外部参考基准；本次未下载或运行。 |

[D4] ASSISTmentsData. *2015 ASSISTments Skill Builder Data*. https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data  
[D5] Choi, Y., et al. (2020). *EdNet: A Large-Scale Hierarchical Dataset in Education*. https://github.com/riiid/ednet  
[D6] PSLC DataShop. *KDD Cup 2010: Educational Data Mining Challenge*. https://pslcdatashop.web.cmu.edu/KDDCup/  
[D7] Liu, Z., et al. (2023). *simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing*, Appendix A. https://arxiv.org/html/2302.06881v2

### 受控数据存储位置（2026-08-27 最终更新）

为避免原始学生记录进入网页项目的版本保存、构建目录或任何Git工作树，所有原始ASSISTments文件、学生级预测和私有实验输出已从项目目录迁移至`/home/ubuntu/controlled-research-data/kt-slm-audit/`。实验入口接受环境变量`KT_AUDIT_DATA_ROOT`，默认值仅为本地复现便利而设；当前项目实际运行时明确设置该目录外路径。该移动不改变文件内容、来源SHA-256或公开聚合结果。
