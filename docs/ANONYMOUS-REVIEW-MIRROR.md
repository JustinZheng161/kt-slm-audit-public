# 双盲审稿匿名镜像发布说明

本仓库是**开发与审计源仓库**，其 GitHub URL、所有者、提交者、issue、release、网页链接和历史记录可能泄露作者身份。因此，投稿文稿不得直接引用此开发仓库 URL。

## 审稿前的必需流程

使用一个支持身份剥离的只读镜像服务（例如 [Anonymous GitHub](https://anonymous.4open.science/)）从已验证的公开复现仓库创建镜像。该服务说明其镜像用于双盲同行评审，并允许配置来源仓库、待删除术语和到期时间。[1]

| 步骤 | 必须操作 | 完成判据 |
| --- | --- | --- |
| 1 | 选择固定的公开仓库提交，并记录其短 SHA。 | 镜像与准备投稿的可复现代码完全一致。 |
| 2 | 在镜像服务中配置所有作者姓名、实验室/机构、邮箱、GitHub 用户名、资助编号和项目别名为待删除术语。 | 镜像的 README、代码注释、文档、图件元数据和网页链接中无上述标识。 |
| 3 | 设定覆盖审稿周期的到期时间，并以无登录浏览器打开镜像链接。 | 无登录访问可查看 README、代码、聚合结果和图件。 |
| 4 | 用固定的 README 命令进行一次独立复现性检查。 | 命令能读取获得许可后放入受控根目录的原始数据，并只写出聚合安全工件。 |
| 5 | 仅在上述检查完成后，将实际 `https://anonymous.4open.science/r/<random-slug>` 填入投稿稿的 Data and Code Availability 段。 | 文稿不含开发仓库 URL 或任何可识别作者身份的 URL。 |

> **禁止项：** 不得上传或镜像原始 CSV、学生 ID、学生级划分、学生序列、预测、每技能参数、bootstrap replicate 值、模型检查点或任何可逆派生数据。即使镜像服务自称私密，也必须遵守此发布边界。

## 当前文稿占位符

在真实匿名镜像创建并用无登录浏览器验证前，文稿使用 `[ANONYMOUS_REVIEW_CODE_URL]` 占位符。该占位符不是有效链接，不能被解释为已提供匿名访问。

## 审稿人复现入口

经审核的匿名镜像应保留根目录的 `README.md` 与以下命令：

```bash
python experiments/run_student_disjoint_kt.py --dkt-epochs 8 --threads 4
python experiments/run_clean_seed_check.py
python experiments/run_label_noise_robustness.py
python experiments/run_revision3_extended_evidence.py
python analysis/analyze_revision3_paired_metrics.py
python analysis/summarize_seed_observed_ranges.py
python analysis/profile_history_length_distribution.py
python experiments/run_window_length_sensitivity.py
python analysis/run-revision6-statistical-and-bkt-audits.py
python analysis/make_paper_figures.py
python analysis/make_revision5_window_figure.py
python analysis/make-revision6-bkt-diagnostic-figure.py
```

其中涉及官方原始数据的命令需要已获许可的用户将文件仅放入仓库外的 `KT_AUDIT_DATA_ROOT`。所有提交到 GitHub 的结果必须先通过敏感路径扫描。

## 参考来源

[1] [Anonymous GitHub — “Share the code, not the author”](https://anonymous.4open.science/)。
