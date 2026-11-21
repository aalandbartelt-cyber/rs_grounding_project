# 遥感文本驱动视觉定位：本地可信评测 v2

工作位置：`D:\计算机视觉\rs_grounding_eval`。数据来源：同级目录 `rs_grounding_train_v1`。本工具不修改训练包，也不访问或修改 GitHub 仓库。

本次是从本地训练包重新构建的 v2；沿用之前 dev4 的 346/30/51 思路，但未恢复旧压缩包中的具体划分，不应把旧版本分数与本版本直接比较。当前名单和源文件 SHA256 已冻结。

## 1. 先理解评测

检索：输入一句描述，从 81 张图片中返回至少 5 个图片 ID，按匹配程度排序。

定位：输入图片及描述，输出一个 `[x,y,w,h]` 框。`w,h` 是宽高，不是右下角坐标。

评分依据是训练包原始 README，副本在 `docs/DATASET_README.md`：

`总分 = 0.1 × R@1 + 0.2 × R@5 + 0.6 × Acc@0.5 + 0.1 × Acc@0.75`

内部指标和 `total` 在 0～1；`total_percent` 在 0～100。R@1/R@5 衡量正确图片是否进入前 1/5 名；Acc 衡量预测框与真值框的交并比 IoU 是否达到阈值。恰好等于 0.5/0.75 算命中。

这里实现的是依据已提供规则构建的本地代理评测。平台私有评分脚本、实际查询生成方式和完整容器接口尚未取得，不能保证与平台隐藏实现完全相同。

## 2. 固定数据划分

| 用途 | 图片数 | 目标数 | 查询数 | 检索图库 |
|---|---:|---:|---:|---:|
| 训练 `splits/train_images.json` | 346 | 1913 | 不生成评测查询 | 不适用 |
| 主要开发评测 `dev4_81` | 30 | 612 | 1224 | 81 张 |
| 补充评测 `official_val51_81` | 51 | 191 | 382 | 同一批 81 张 |

每个目标各生成一条检索和一条定位查询。81 张图库 = 30 张开发图 + 51 张官方验证图，两套协议的查询图片不同。`official_val51_81` 是本地协议名，并非官网原始验证协议。

官方验证图片没有车辆类，因此额外从原 train 留出 30 张覆盖四类的开发图。按主要目标类别分桶，配额为油罐 9、舰船 9、车辆 3、飞机 9 张；SHA256(seed/key) 排序选取，混合类别按目标数量最多的类别分桶，并列取较小 category_id。完整名单、固定 seed 和全部源文件指纹见 `splits/manifest.json`。

后续训练必须按 `splits/train_images.json` 读取 346 张图；其中路径相对 `rs_grounding_train_v1`。不要直接扫描原 train 的 376 张图片，也不要把 81 张留出图片的标注加入训练、伪标签、样本缓存或描述检索库。冻结一次划分后，模型之间使用相同训练名单和图库。开发集用于调参；补充验证集尽量只在关键里程碑使用。若反复利用其分数调参，它也会成为开发集。

主分按查询平均，开发集车辆 458 个目标占比较高；必须同时查看 `by_category`。缺失类别显示 `null`，不冒充零分。`macro_category_total_diagnostic` 为有样本类别的平均，仅供诊断，不替代官方权重总分。

## 3. 文件位置与隔离

| 路径 | 用途 |
|---|---|
| `evalkit.py` | 构建协议、严格检查预测格式、计算指标 |
| `protocols/dev4_81/input_path/` | 模型可读取的图片、info.json、queries.json |
| `protocols/official_val51_81/input_path/` | 补充协议的公开输入 |
| `private/*_gt.json` | 仅评分程序可读的答案；不要传给模型 |
| `splits/` | 冻结划分及训练文件清单 |
| `audit/` | 数据审核、构建摘要、图片相似性审核和自检结果 |
| `baselines/` | 读取真值的评分自检夹具；oracle/zero 不能作为参赛模型 |
| `blind_baseline.py` | 只读公开输入的极弱基线，展示离线输出接口 |
| `runs/blind_random/` | 真实走完输入→预测文件→评分链路的随机基线结果 |
| `test_evalkit.py` | 32 项手算指标、边界和异常格式测试 |

`info.json` 本地采用数组，元素为 image_id/file_name/width/height。图片 ID 仅在当前冻结协议中有效，与原图片文件名无关；模型必须从 info.json 读取映射。检索查询没有答案图片 ID；定位查询只提供任务所需的 image_id。

## 4. 现在就可以运行

在 PowerShell 中：

```powershell
Set-Location 'D:\计算机视觉'
.\rs_grounding_eval\check_all.ps1
```

这会重建并校验固定名单、运行测试及满分/零分检查、生成不读答案的随机预测、评分，并确认原始训练包 1284 个文件未变。需要 Python 3.10+，不依赖 torch、numpy、Pillow 或 pytest，无需联网。若本机策略阻止 ps1，可逐条执行其中的 Python 命令。

重新做图片解码及相似性筛查：

```powershell
.\rs_grounding_eval\check_all.ps1 -ImageAudit
```

可选图片筛查使用本机已验证的 Windows PowerShell 环境中的 System.Drawing；打包的 JSON 审核结果可直接阅读，无需重新运行。该筛查不是跨平台依赖。

## 5. 接入自己的模型

将主要协议的 `input_path` 传给模型，让它输出 `pred.json`。输入里仅包含图片和查询，不要把整个评测目录挂给模型。可先运行示例：

```powershell
python .\rs_grounding_eval\blind_baseline.py --input_path .\rs_grounding_eval\protocols\dev4_81\input_path --output_path .\rs_grounding_eval\runs\my_example --mode random
python .\rs_grounding_eval\evalkit.py score --gt .\rs_grounding_eval\private\dev4_81_gt.json --pred .\rs_grounding_eval\runs\my_example\pred.json --out .\rs_grounding_eval\runs\my_example\metrics.json --require-complete
```

换模型时保留评分命令，只替换预测文件。补充协议将路径中的 `dev4_81` 换为 `official_val51_81`。

每条查询输出一个对象：

```json
[
  {"query_id": 1, "type": "retrieval", "ranking": [1, 2, 3, 4, 5]},
  {"query_id": 2, "type": "grounding", "image_id": 1, "category_id": 2, "score": 0.93, "bbox": [989, 487, 20, 47]}
]
```

以上仅是格式示例，不代表当前第 1/2 条查询的正确答案。类别固定为 1 油罐、2 舰船、3 车辆、4 飞机。

评分报告包含主分、每类别分数、逐查询 IoU/命中、漏答数量、图片级 bootstrap 95% 区间，以及输入预测/真值/评分器 SHA256。默认 bootstrap 1000 次，按源图片重采样，避免把一张图片的数百个目标当作独立样本；区间不代表线上分布差异或训练随机性的影响。可用 `--bootstrap 0` 关闭。

## 6. 明确的本地评分约定

- JSON 四角点取 min/max，保留原始浮点值，统一转换为 xywh；不取整、不裁剪、面积计算不加 1。
- 每条查询只认原始标注的一张图/一个框。重号、未知 ID、错误任务类型、非法类别、非有限数、非正宽高、重复排序图片和不足 Top-5 的记录直接报错，不生成新成绩。漏答默认计零；`--require-complete` 另外要求全部回答。
- 训练包只将定位准确率定义为 IoU，未明说类别错误是否另外扣分，因此主定位分按 IoU；额外提供 `category_gated_diagnostic`，显示必须类别一致时的分数。类别字段本身仍严格要求 1～4。
- 重复描述导致多答案歧义。主分仍使用原始单答案，`multi_target_diagnostic` 额外接受同一规范化描述且同类别的其他图片/同图框；这是放宽诊断，不能当作官网成绩。规范化仅用于诊断，公开查询保留原文。
- `score` 字段校验在 0～1，但一条定位查询仅有一个框，当前 IoU 指标不用置信度重排或加权。

上述未被训练包细化的处理是显式本地约定；拿到官方评分脚本后应逐项做对照，不默认为已验证的官网行为。

## 7. 已知数据问题与当前结论

详细结论见 `REPORT.md`。主要问题是验证集缺车辆、描述重复、19 张图尺寸非 1024×1024、109 张 JSON/XML 坐标存在差异，以及 1 个 JSON 框右边缘略越界。由于描述与 JSON 框配对，评测使用 JSON；原始数据保持不动。

目前可开始接入第一个真实模型，先运行不训练的基线，再仅用 346 张图微调。不要把 oracle 满分或随机基线当作模型能力结果；当前还未训练或评测任何真实视觉模型。
