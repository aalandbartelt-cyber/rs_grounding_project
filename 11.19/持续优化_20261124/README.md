# 11.19 后续融合实验：2026-11-24

当前保留内部开发总分 **0.5259/1**，尚未达到 0.6。仅使用 49 张公开原始训练图，33 张拟合、16 张内部开发；没有读取官方验证或测试图像，也没有用旧 47 张留出图训练或评分。

- [逐轮总分与核验](fusion_train_only/optimization_results/README.md)
- [分步运行说明和实际结果](fusion_train_only/复现步骤_20261124.md)
- [当前结果索引](fusion_train_only/optimization_results/current_best.json)
- [数据访问和哈希核验](fusion_train_only/optimization_results/verification.json)

本目录保存代码、日志与实验记录。图像、模型大权重和完整中间特征缓存保留在 E 盘原项目，不包含在本归档中。它不是自包含训练包、Docker 镜像或通过 0.6 门槛的权重导出包。本轮设备为 RTX 4060 Laptop；A100 本轮未实测，原 A100 环境记录继续单独归档。

[下载本轮代码与记录压缩包](11.19_持续优化_20261124.zip) · [SHA-256](11.19_持续优化_20261124.zip.sha256)
