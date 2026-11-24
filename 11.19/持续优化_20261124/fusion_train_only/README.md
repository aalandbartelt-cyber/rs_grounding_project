# 遥感文搜图：仅原始训练图的融合实验

当前保留方案的内部开发总分为 **0.5259259259 / 1**，尚未达到 0.6。方案为 v2 检索头与 v1 RemoteCLIP 视觉 LoRA 定位；固定的 16 张原始训练内部开发图上，每项任务各有 27 条查询。

| R@1 | R@5 | Acc@0.5 | Acc@0.75 | 总分 / 1 |
|---:|---:|---:|---:|---:|
| 5/27 | 15/27 | 16/27 | 11/27 | 0.5259259259 |

总分 = 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75 = 142/270。

逐轮分数、模型选择与核验入口：[optimization_results/README.md](optimization_results/README.md)。v13–v18 没有通过拟合折筛选，因此这些轮次的总分标记为沿用既有 0.5259，不能称为新增开发评测。

数据仅为已隔离的 49 张公开原始训练图，33 张用于拟合、16 张用于内部开发。相同 SIFT 场景重叠组不跨边界。官方验证集、测试集及旧 47 张留出图不进入本实验。内部开发分数此前已参与架构迭代，属于开发成绩。

项目固定在 `E:\ChatGPT\智元`；PowerShell 先执行 `. .\启用E盘环境.ps1`，再使用 `fusion_train_only\.venv\Scripts\python.exe`。实验需要本地的训练白名单、隔离训练包、预训练特征缓存、各轮中间推理特征和预训练模型。代码归档不包含这些完整输入，不能单凭归档直接从零复现全部轮次。

本轮设备为 RTX 4060 Laptop 8 GB，未在 A100 重跑。新下载的 DINOv2 与 SigLIP 2 权重位于本目录 `weights/`；其固定版本和 SHA-256 见 `training_v15/results/dino_features_manifest.json` 和 `training_v18/results/siglip2_features_manifest.json`。新下载的是通用预训练权重，未达到分数门槛，未导出正式融合权重包或提交镜像。

运行步骤和预期结果见 [复现步骤_20261124.md](复现步骤_20261124.md)。

最终提交的 Docker 镜像导出包（`.tar` 或 `.tar.gz`）不得超过 **10 GB，即 10,000,000,000 字节**，按最终文件的实际大小验收。限制见 [submission_limits.json](submission_limits.json)；导出后运行 `python fusion_train_only/code/check_submission_size.py <镜像包路径> --report <检查报告路径>`，超限时返回非零退出码。正式镜像尚未生成。
