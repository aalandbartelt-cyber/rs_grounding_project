# 遥感图像文搜图定位挑战赛

> 赛期：2026-11-01 ~ 2026-12-14 23:59
> 评测：总分 = 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75
> 赛事主页：http://121.196.102.212/portal/track/1856158738000052230

## 任务

- **任务 A 图像检索**：给定目标描述文本，从评测图库（81 张 1024×1024 遥感大图）中输出图像排序
- **任务 B 区域定位（grounding）**：给定图像 + 描述文本，框出所描述目标

## 数据

`data/rs_grounding_train_v1/`

| 项 | 数量 |
|---|---|
| train images / labels_json / labels_xml | 376 / 376 / 376 |
| val images / labels_json / labels_xml | 51 / 51 / 51 |
| 描述-框监督对 | 3095 |

- 标注：labelme JSON（含 `description` 中文描述 + `points` 矩形框）、VOC XML（无描述）
- 类别（category_id）：1 油罐、2 舰船、3 车辆、4 飞机
- 原始官方包：`data/rs_grounding_train_v1.zip`（135M）

## 提交

Docker 镜像包（tar/tar.gz，≤10GB），评测容器**断网**、独占 1×A100-80GB、限时 60 分钟。

```
/input_path/                images + info.json + queries.json
queries.json 的 type 字段：
  retrieval  -> {"query_id", "type": "retrieval", "text"}
  grounding  -> {"query_id", "type": "grounding", "image_id", "text"}
/output_path/pred.json
  retrieval  -> {"query_id", "type": "retrieval", "ranking": [image_id, ...]}
  grounding  -> {"query_id", "type": "grounding", "image_id",
                 "category_id": 1, "score": 0.9, "bbox": [x, y, w, h]}
```

依赖与权重必须打包进镜像；禁止使用评测集图像、运行时联网、硬编码答案。

## 目录

```
rs_grounding_project/
├── data/            # 数据集 + 官方 zip（不追踪大文件时按需 .gitignore）
├── docs/            # 赛题说明（docx/pdf）
├── src/             # 训练/推理代码
├── submission/      # Dockerfile / run.py / weights
└── README.md
```

## 技术路线参考

- 检索：Chinese-CLIP / BLIP-2 图文相似度，图像侧可切片或多尺度
- 定位：GroundingDINO / MDETR / OWL-ViT，或检测器 + 文本匹配两阶段
- 微调：LoRA / 量化，3095 条描述-框对；可用 DIOR-R、RSICD 公开数据预训练
