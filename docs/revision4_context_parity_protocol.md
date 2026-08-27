# Revision 4：历史窗口一致性审计协议

## 目的与证据状态

本协议补充 Revision 3 的归档实验，而不替代其主结果。Revision 3 的 DKT 训练将每位学生历史切分为最长 200 个 transition 的非重叠窗口，但归档验证与测试阶段向模型提供完整学生历史。这个差异可能影响模型可用上下文，因而应当被显式审计，而不应被隐含在实现细节中。

当前仓库提供了 `experiments/run_context_parity_audit.py`。**该脚本在本次修订时尚未使用受控源文件执行，因此没有新的 AUC、校准指标或图表被报告。** 该约束是为了避免把未运行的设计写成实际实验结果。

## 预先固定的比较

| 项目 | 固定规则 |
| --- | --- |
| 数据源与划分 | official corrected ASSISTments2009 collapsed CSV；固定学生级 80/10/10 划分；split seed `20260822`。 |
| 模型 | DKT-64；embedding/hidden dimension 均为 64；dropout=0。 |
| 优化 | AdamW，`weight_decay=0`，`lr=0.002`，batch size=64，gradient-norm clip=5.0。 |
| 训练预算 | 最多 8 epoch；仅以 validation ROC-AUC 选择 checkpoint；每个训练运行只在选择后对测试集评估一次。 |
| 随机性 | 固定种子集合：`20260822`、`20260823`、`20260824`。 |
| 训练窗口 | 每一训练样本最多包含 200 个 transition；窗口不跨越学生边界。 |
| 对照 A | `full_student_history_legacy`：验证/测试使用完整学生历史；这是归档 Revision 3 的协议。 |
| 对照 B | `matched_train_chunks`：验证/测试用与训练相同的非重叠 200-transition 窗口。 |
| 指标 | ROC-AUC、Brier score、固定十等宽 bin 的 ECE；所有指标基于同一批第二次及以后测试交互。 |

## 执行与安全输出

在拥有官方数据访问许可的受控环境中，设置数据根目录并运行：

```bash
export KT_AUDIT_DATA_ROOT=/absolute/path/to/controlled-data
python experiments/run_context_parity_audit.py
```

脚本要求两种协议都对每个测试学生的第二次及以后交互**恰好计数一次**。它仅在 `results/revision3/revision4_context_parity_audit.json` 写入每种子的聚合指标、协议元数据及成对差异。原始 CSV、学生标识、训练/验证/测试成员、序列、逐行预测和检查点只保留在 `KT_AUDIT_DATA_ROOT`，不应被提交到任何仓库。

## 报告规则

结果应作为单独的“上下文可用性敏感性审计”小节呈现，不能回溯性替换 Revision 3 的主结果。三种子差异只能描述为观测到的配置特定差异；在没有预先指定最小效应阈值、更多独立种子与更广泛数据集复现前，不得表述为等效、无效或跨数据集结论。
