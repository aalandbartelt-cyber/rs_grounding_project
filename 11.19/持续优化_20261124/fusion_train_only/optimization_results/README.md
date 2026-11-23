# 连续优化记录与当前保留结果

当前最佳为 **0.5259/1**，方案：v2 新检索 + v1 LoRA 定位（保留）；尚未达到 0.6 目标。所有结果采用同一 16 张原始训练内部开发图、每任务 27 条查询计算，不能称为官方测试成绩。

总分 = 0.1×R@1 + 0.2×R@5 + 0.6×Acc@0.5 + 0.1×Acc@0.75。

| 轮次 / 方案 | R@1 | R@5 | Acc@0.5 | Acc@0.75 | 总分 / 1 |
|---|---:|---:|---:|---:|---:|
| 首轮固定五框架融合 | 3/27 | 8/27 | 16/27 | 11/27 | 0.4667 |
| v1 监督融合头 | 3/27 | 13/27 | 16/27 | 10/27 | 0.5000 |
| v1 RemoteCLIP 视觉 LoRA | 3/27 | 13/27 | 16/27 | 11/27 | 0.5037 |
| v2 多视角、配对、属性文本全组合 | 5/27 | 15/27 | 15/27 | 10/27 | 0.5000 |
| v2 新检索 + v1 监督定位 | 5/27 | 15/27 | 16/27 | 10/27 | 0.5222 |
| v2 新检索 + v1 LoRA 定位（保留） | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v3 区域排序与框回归候选 | 5/27 | 15/27 | 14/27 | 10/27 | 0.4778 |
| v4 中文 CLIP 与语义相容监督全组合 | 5/27 | 15/27 | 15/27 | 11/27 | 0.5037 |
| v4 新检索 + 保留 LoRA 定位 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v5 GeoRSCLIP QuickGELU 对照全组合 | 5/27 | 15/27 | 15/27 | 12/27 | 0.5074 |
| v5 新检索 + 保留 LoRA 定位 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v6 独立飞机属性证据全组合 | 2/27 | 11/27 | 15/27 | 10/27 | 0.4593 |
| v6 新检索 + 保留 LoRA 定位 | 2/27 | 11/27 | 16/27 | 11/27 | 0.4852 |
| v7 几何修正与同物体框融合 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v8 成对检索 | 5/27 | 13/27 | 16/27 | 11/27 | 0.5111 |
| v9 扩展 LoRA 候选框 | 5/27 | 15/27 | 15/27 | 12/27 | 0.5074 |
| v10 Qwen 图库图像比较 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v11 LoRA 秩与层数搜索 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |
| v12 查询提示交叉验证 | 5/27 | 15/27 | 16/27 | 11/27 | 0.5259 |

未晋级的轮次只完成拟合折筛选；下表总分沿用已核验的既有最佳，不是新增评测。定位加权分只计算 0.6×Acc@0.5 + 0.1×Acc@0.75，最高为 0.7。

| 轮次 | 最佳拟合折定位加权分 / 0.7 | 晋级基线 / 0.7 | 当前沿用开发总分 / 1 |
|---|---:|---:|---:|
| v13 成对 LoRA 损失 | 0.4574 | 0.4593 | 0.5259（沿用） |
| v14 非线性候选排序 | 0.4741 | 0.4852 | 0.5259（沿用） |
| v15 DINOv2 + RemoteCLIP 直接融合 | 0.4722 | 0.4852 | 0.5259（沿用） |
| v16 查询条件化 DINOv2 + CLIP | 0.4833 | 0.4852 | 0.5259（沿用） |
| v17 更长训练的候选排序 LoRA | 0.4315 | 0.4852 | 0.5259（沿用） |
| v18 SigLIP 2 定位与检索融合 | 0.4611 | 0.4852 | 0.5259（沿用） |

## 数据与模型使用

源数据仅为本地隔离的 49 张公开原始 train 民航图。33 张图用于拟合，16 张用于内部开发评分；SIFT 重叠场景不跨界。各轮超参数使用 33 张内的场景分组三折选择，模型和参数在开发评分前保存。开发分数被反复用于架构迭代和保留方案，因此这里是开发成绩。折内较小图库的交叉验证分数只用于选择参数，不替代本表。

模型输入来自查询文本、图像与预测框。没有使用官方验证集、测试集或旧 47 张留出图；没有按文件名、查询 ID 或真值编写预测。独立属性分类只读取预测区域，不把查询或答案送进属性提示。检索的属性聚合覆盖图库中每张图的检测候选，没有使用定位任务提供的正确图像映射。

实际设备为本机 RTX 4060 Laptop 8 GB。大视觉语言模型使用 NF4 4 位量化，推理进程通过审计钩子阻止联网；本轮没有声称已在 A100 上重跑。大模型顺序加载。

## 每轮改动与复现

1. `training_v2`：GeoGround 旋转框、Qwen 双向候选比较、实际输出框的区域编码、文本属性提示、Qwen 分数去饱和。依次运行 `code/geoground_views.py`、`code/qwen_pairs.py`、`code/encode_candidates.py`、`code/encode_prompts.py`、`code/train_round.py`、`code/retain.py`。
2. `training_v3`：冻结视觉向量与查询交互、查询内中心化排序、候选差分训练、框回归。运行 `code/region_ranker.py`。三折未选择视觉交互或框回归，保留了完整负结果。
3. `training_v4`：中文 CLIP 证据、原始/扩展候选对照、已训练头与新头的折外融合、按查询属性构造语义相容监督。运行 `code/train_fusion.py`。
4. `training_v5`：GeoRSCLIP 的 QuickGELU 激活函数对照，图像和文本同步重新编码。先运行 `code/encode_activation.py`，再运行 `code/train_activation.py`。官方示例不足以断言 L/336 先前的激活配置错误，因此只作为实验对照报告。
5. `training_v6`：在预测区域上独立提取发动机数、涂装、飞机完整度；先运行 `code/extract_attributes.py`，再运行 `code/train_attributes.py`。同一物体只缓存预测证据，没有缓存标签答案。
6. `training_v7`：用拟合图学习紧凑的框几何修正，并比较同一物体的多模型框融合。运行 `code/refine_boxes.py`。校准只在拟合图的场景分组折内训练，开发图仍用已保存的 LoRA 选框器。
7. `training_v8`：成对比较检索候选并用折内检索指标筛选，运行 `code/pairwise_retrieval.py`；开发总分为 0.5111。
8. `training_v9`：扩展 LoRA 定位候选框，运行 `code/score_expanded_lora.py`；开发总分为 0.5074。
9. `training_v10`：比较 Qwen 图库图像证据，运行 `code/encode_gallery_pairs.py`、`code/train_gallery_ranker.py` 和 `code/test_gallery_ranker.py`。1,204 组成对比较已归档并通过访问检查；最终总分与现有最佳相同，为 0.5259。
10. `training_v11`：在 33 张拟合图内搜索 LoRA 秩和解冻层数，运行 `code/cv_lora.py`、`code/score_lora_cv.py`；没有超过现有开发总分。
11. `training_v12`：在拟合折内比较查询提示模式，运行 `code/lora_prompt_cv.py`；没有超过现有开发总分。
12. `training_v13`：在预测框上用图文二元对齐与成对硬负例排序混合训练 LoRA，运行 `code/pairwise_lora.py`。四档成对损失权重、三折场景分组交叉验证的最佳加权定位分未超过 0.4593 基线，因此没有重评内部开发集或保存替代权重；详情见 `training_v13/results/selection.json`。
13. `training_v14`：在相同候选池上使用 ExtraTrees、HistGradientBoosting 预测 IoU/定位效用；三折最佳 0.4741，未超过已复现的 v3 场景折定位基线 0.4852。运行 `code/tree_ranker.py`。
14. `training_v15`：下载 Apache-2.0 的 `facebook/dinov2-base`，在 518×518 的紧框与 1.5 倍上下文上提取 CLS 和平均 patch 特征，再与 RemoteCLIP 折内适配器结合。运行 `code/encode_dino_regions.py` 与 `code/dino_ranker.py`；三折最佳 0.4722，未晋级。
15. `training_v16`：将 DINOv2 区域外观核与 CLIP 查询语义核相乘，让排序依赖查询文本；运行 `code/conditional_kernel.py`。三折最佳 0.4833，未超过 0.4852 基线，保留原方案。
16. `training_v17`：固定候选池，使用查询内定位效用分布训练候选排序 LoRA；学习率 0.0005/0.002，在 6/12 个 epoch 比较折外定位分。运行 `code/listwise_lora.py`；最佳 0.4315，未超过 0.4852 拟合折基线，未进入开发集评分。
17. `training_v18`：对照多语言 SigLIP 2 的中文、英文及融合文本特征，分别比较候选框排序与全图库检索。运行 `code/encode_siglip2.py`、`code/train_siglip2.py`。定位最佳 0.4611/0.7，检索最佳 0.1630/0.3，均未超过拟合折基线；没有新增开发评分。

在项目根目录使用 `fusion_train_only/.venv/Scripts/python.exe` 调用上述完整脚本路径。实验依赖先前的预训练特征和模型文件；此处是实验复现入口，未作为接收任意新图像的正式提交接口。

## 结果和权重

`round_scores.json` 保存首轮和 v1–v12 的同协议开发总分和命中数。`fit_only_cv_rounds.json` 保存 v13–v18 的筛选分数与沿用总分。`current_best.json` 是当前结果索引，`best_predictions.json` 保存逐条开发预测，`verification.json` 记录分数验算、原始训练图白名单、场景分离和代码/权重哈希检查。v13–v18 没有独立开发成绩，因为它们没有通过拟合折筛选。

新增 DINOv2 预训练权重存放于 `weights/DINOv2-base/model.safetensors`（346,345,912 字节），固定版本 `f9e44c814b77203eaa57a6bdbbd535f21ede1415`，SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`。这是通用预训练权重；本轮没有生成通过 0.6 分门槛的融合权重或正式镜像包。模型来源、特征哈希及数据范围见 `training_v15/results/dino_features_manifest.json`。

新增 SigLIP 2 预训练权重存放于 `weights/SigLIP2-base-512/model.safetensors`（1,503,344,520 字节），固定版本 `a89f5c5093f902bf39d3cd4d81d2c09867f0724b`，SHA-256 `fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe`。模型和 tokenizer 已完成本机离线加载与原始训练图特征提取；这不代表正式 A100 容器实测。

拟合数据诊断：54 条查询只包含 18 种不同文字，4 个图像—文字组合含多个目标。同一文字固定一个排序时，33 张拟合图库的 R@1 上限为 0.3889；同一图像—文字组合只能输出一个框时，当前候选池可实现的拟合定位加权上限约 0.6352。这里的上限通过拟合标签事后计算，只解释数据歧义，不属于模型得分，也不用于输出预测。详情见 `fit_only_diagnostics.json`。

当前保留方案的检查点从 `training_v2/results/retained_result.json` 追溯。若方法为 `v2_retained`，使用 `training_v2/weights/multiview_candidate_ranker.joblib` 内的检索头和 `training_v1/weights/RemoteCLIP_visual_lora.pt`，并通过 `training_v2/code/retain.py` 复现；它们还依赖上一级 `weights/` 中的预训练网络。

实验参数已保存，正式权重导出包和 Docker 镜像包未生成。验算脚本为 `optimization_results/code/finalize.py`。
