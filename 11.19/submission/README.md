# 遥感文搜图定位 · 民航客机提交镜像

`rs-grounding-civilian-v1.tar.gz` 是 Linux/amd64 Docker 镜像归档，镜像标签 `rs-grounding-civilian:v1`。镜像内附 Python 依赖、Chinese-CLIP、GroundingDINO 和本轮采用的两个投影权重。当前镜像使用 CPU 版 PyTorch。

## 赛题接口

平台执行 `python /workspace/run.py`。程序从只读 `/input_path/` 读 `images/`、`info.json` 和 `queries.json`，将 JSON 数组写到 `/output_path/pred.json`。检索输出 5 个按相似度排序的图库 ID。定位输出原图像素 `[x, y, w, h]`、飞机类别 `4` 与检测置信度。`info.json` 支持图片记录列表、包含 `images` 列表的对象，以及按图片 ID 索引的映射。记录用 `image_id` 或 `id` 指定 ID，并用 `file_name` 等字段指向图片。字段缺省时按不带扩展名的图片文件名查找。PDF 未提供 `info.json` 样本，平台的实际清单格式仍待核对。

本镜像覆盖经复核的民航客机任务。范围外的检索查询返回空列表，范围外或未检出候选的定位查询返回零置信度和空框。PDF 要求检索至少返回 Top 5，完整混合评测集可能因这些弃权记录无法通过平台格式校验。

## 导入与断网试跑

在支持 Linux 镜像的 Docker 主机上准备民航样例目录 `input/`，从本目录执行：

```bash
docker load -i rs-grounding-civilian-v1.tar.gz
mkdir -p output
docker run --rm --network none --shm-size=2g \
  -v "$PWD/input:/input_path:ro" \
  -v "$PWD/output:/output_path" rs-grounding-civilian:v1
```

输出为 `output/pred.json`。正式评测环境为单张 A100-80GB，容器断网，共享内存 2 GB，限时 60 分钟，提交镜像归档不超过 10 GB。

如需按 Dockerfile 重新构建和导出，从本目录运行：

```bash
python prepare_weights.py rs-grounding-civilian-v1.tar.gz
docker build --platform linux/amd64 -t rs-grounding-civilian:v1 .
set -o pipefail
docker save rs-grounding-civilian:v1 | gzip -1 > rs-grounding-civilian-rebuilt.tar.gz
sha256sum rs-grounding-civilian-rebuilt.tar.gz
```

权重提取脚本先核对发布附件的 SHA-256，再提取 `weights/`。已经准备好权重时可以直接构建。构建需要联网安装依赖，运行时模型和分词器均从镜像内加载。

## 已验证结果

正式平台截图显示总分 **0.1420 / 1**，用时 3 分 31 秒。四项指标与原图收录在 [11.19 平台成绩](../README.md#正式平台成绩)。

本地执行镜像内同一 `run.py`，使用 47 张民航留出图生成 150 条预测，耗时 169.86 秒。按赛题公式复算总分 **0.4760 / 1**。R@1、R@5、Acc@0.5、Acc@0.75 的命中数分别是 7/75、12/75、47/75、44/75。

已核对本地归档的镜像入口、平台架构、离线配置、归档层哈希和模型文件。本机缺少 Docker 服务，本地 `docker load` 与断网容器试跑仍待验证。

离线复核使用公开训练包中已复核的 7 张民航图，清空模型缓存并在本机阻断 Python 的 IP 套接字和 DNS 调用。检索和定位均生成有效结果，输入文件哈希与运行前一致。依赖初始化时仅有一次本地 IPv6 回环能力探测，被测试钩子拦截后推理继续。没有发现连接外部服务或解析域名的尝试。镜像层中没有预装比赛图像、查询、标注或预测文件。支持范围内的排名和坐标由模型根据当次挂载的输入实时计算。
