# 遥感图像文本驱动视觉定位挑战赛 — 训练数据包说明

> 欢迎参赛！本数据包为官方训练数据，请先阅读本说明再开始。

## 一、数据包内容

```
train/                       # 训练集 376 张
val/                         # 验证集 51 张（可自行划分训练/调参）
  └── 每个子集均含：
      images/*.jpg           #   遥感图像（1024×1024）
      labels_json/*.json     #   labelme 标注（矩形框 + 目标中文描述）
      labels_xml/*.xml       #   VOC XML 标注（与 JSON 内容一致，无描述字段）
classes.txt                  # 类别表（中文，顺序即 category_id）
label_map.json               # 类别 → category_id 映射
README.md                    # 本说明
```

**目标类别**（category_id 提交时按下表）：

| category_id | 类别 |
|---|---|
| 1 | 油罐 |
| 2 | 舰船 |
| 3 | 车辆 |
| 4 | 飞机 |

## 二、任务说明（重要）

本赛题为**文本驱动的视觉定位**，包含两个子任务：

- **任务A 图像检索（retrieval）**：给定一段目标描述文本，从评测图像库中找出最匹配的图像（输出排序）
- **任务B 区域定位（grounding）**：给定图像 + 目标描述文本，在图中定位所描述的目标，输出检测框

标注中每个目标的 `description` 字段即描述文本（如"这是一艘灰色涂装的中型舰船，舰体呈长条形……"），
它与 `points` 矩形框一一对应——这就是"文本 → 框"的训练监督信号。

**标注格式速览**（labelme JSON）：
`shapes[].points` 为矩形两角点（左上/右下），`shapes[].description` 为该目标的自然语言描述。

## 三、评测说明（重要）

- 评测集由平台持有，**不在本包内**（81 张图，场景分布与训练集一致）
- 评测输入：`/input_path/`
  - `images/` 评测图像
  - `info.json` 图像元信息（image_id / file_name / width / height）
  - `queries.json` 查询列表，每条形如：
    - `{"query_id": 1, "type": "retrieval", "text": "一座白色圆形储罐……"}`
    - `{"query_id": 2, "type": "grounding", "text": "静止停泊的灰色舰船……", "image_id": 17}`
- 提交方式：**Docker 镜像包**（tar/tar.gz，含模型权重，≤2GB），镜像规范见《算法镜像包提交规范》
- 运行环境：评测容器**断网**、独占 1 张 NVIDIA A100-80GB、限时 60 分钟
- 输出：`/output_path/pred.json`，条目格式：

```json
[
  {"query_id": 1, "type": "retrieval", "ranking": [105, 23, 87, 41, 2]},
  {"query_id": 2, "type": "grounding", "image_id": 17, "category_id": 2,
   "score": 0.93, "bbox": [989.0, 487.0, 20.0, 47.0]}
]
```

- `ranking`：按匹配度从高到低排序的 image_id 列表（至少 Top-5，多给不扣分）
- `bbox` 为左上角 + 宽高 `[x, y, w, h]`；每条 grounding 查询输出**一个**最匹配的框
- `category_id` 按【二】类别表 1~4

**评分指标**：

| 指标 | 含义 | 权重 |
|---|---|---|
| R@1 | 检索排名第一即命中 | 10% |
| R@5 | 检索前五命中 | 20% |
| Acc@0.5 | 定位框与真值 IoU ≥ 0.5 的比例 | 60% |
| Acc@0.75 | 定位框与真值 IoU ≥ 0.75 的比例 | 10% |

总分 = `0.1*R1 + 0.2*R5 + 0.6*Acc05 + 0.1*Acc75`

## 四、建议

- 检索与定位可分开两套模型，run.py 内按 queries.json 的 type 字段自行路由
- 可用公开遥感数据集（DIOR、RSICD、RSVQA 等）预训练，再用本数据微调
- 允许任何公开模型/数据；禁止使用评测集图像、运行时联网、硬编码答案
- 可多次提交，取队伍最高分排名

祝比赛顺利！有问题联系竞赛组委会。
