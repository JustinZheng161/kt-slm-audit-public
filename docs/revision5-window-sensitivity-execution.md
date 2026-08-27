# Revision 5：训练窗口敏感性执行记录

## 目的

该补充分析直接审计一个明确的实施选择：DKT 训练时使用非重叠的最长 200-transition 窗口，而归档主评估使用完整测试学生历史。分析不替换 Revision 3 的主要八 epoch 结果，且不支持跨数据集的窗口长度结论。

## 受控数据来源与边界

用于本地执行的文件来自 ASSISTments 官方 corrected Skill Builder 2009–2010 页面；该页面说明更正文件每行表示 student–problem，多个技能标签被组合到一个条目中。[1] 数据使用条款要求不得向他人提供数据，因此原始 CSV、用户标识、划分成员、序列、逐行预测与检查点一直保留在本地受控目录，未进入本仓库。[2]

| 字段 | 记录 |
| --- | --- |
| 官方数据页面 | `https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010` |
| 校验值 | `SHA-256 162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da` |
| 输出分类 | 仅限 aggregate-safe JSON、图件、脚本和本说明。 |
| 明确排除 | 原始学生数据、student IDs、splits、sequences、predictions、checkpoints。 |

## 固定协议

| 设置 | 固定值 |
| --- | --- |
| 学生级划分 | 固定 80/10/10，split seed `20260822`。 |
| 条件 | 最大训练窗口 200、500 和 full available history（1,028 transition）。 |
| 模型 | DKT-64；embedding/hidden dimension=64；dropout=0。 |
| 优化 | AdamW（weight decay=0），learning rate=0.002，batch size=64，gradient clip=5.0。 |
| 预算/选择 | 最多 8 epoch；仅通过 validation ROC-AUC 选择 checkpoint；每个 run 只在选择后评估一次测试集。 |
| 评价 | 三个种子 `20260822`、`20260823`、`20260824`；所有条件使用同一个完整-history 测试目标集（24,306 个 second-and-later response）。 |

## 聚合结果

| 最大训练窗口 | Mean test ROC-AUC (SD) | 与 200 窗口的配对平均 ΔAUC (SD) |
| --- | ---: | ---: |
| 200 | 0.7654 (0.0011) | Reference |
| 500 | 0.7651 (0.0005) | -0.0003 (0.0010) |
| Full available (1,028) | 0.7651 (0.0002) | -0.0003 (0.0010) |

固定测试划分共有 403 名学生；其历史长度的 P25/P50/P75/P90/P99 分别为 9.5/21.0/56.5/138.2/685.98 个 interaction，其中 22 名（5.5%）历史长度超过 200。窗口分析的观察到的均值差异很小，但三种子设计不支持关于等效性、全局不变性或一般无效的推断。

## 可重复执行

获得官方数据许可后，将原始文件仅放置到外部受控根目录的 `raw/skill_builder_data_corrected_collapsed.csv`，设置 `KT_AUDIT_DATA_ROOT`，并执行：

```bash
python analysis/profile-history-length-distribution.py
python experiments/run-window-length-sensitivity.py
python analysis/make-revision5-window-figure.py
```

脚本仅将供提交的聚合结果写入 `results/revision3/`；所有可逆学生级工件只会写入外部 `KT_AUDIT_DATA_ROOT/results/`。

## 参考来源

[1] [ASSISTmentsData, “Skill-builder data 2009–2010”](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)。

[2] [ASSISTmentsData, “Terms Of Use For Using Data”](https://sites.google.com/site/assistmentsdata/termsofuseforusingdata)。
