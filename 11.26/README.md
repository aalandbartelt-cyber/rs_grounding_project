# 11.26 · CLIP + GroundingDINO + YOLO 车辆融合

本目录记录此前保留的三模型车辆融合版本，内部场景留出实验最高分 **0.5532394366**（epoch 1）。Chinese-CLIP 负责中文描述与区域匹配，GroundingDINO 提供候选框，YOLO26m-OBB 通过滑窗检测车辆，并结合类别路由、颜色/尺寸属性和已训练的排序参数选框。这是车辆专项融合版本；后续三个模型同时联合训练的实验未放入此版本。

## 权重下载

[打开同仓库 11.26 Release](https://github.com/aalandbartelt-cyber/rs_grounding_project/releases/tag/11.26)

| 文件 | 说明 |
| --- | --- |
| [vehicle_fusion_best_0.553239_20261126.tar.gz](https://github.com/aalandbartelt-cyber/rs_grounding_project/releases/download/11.26/vehicle_fusion_best_0.553239_20261126.tar.gz) | 用户指定的原始最优权重归档，1,403,191,296 bytes |
| [SHA256 校验文件](https://github.com/aalandbartelt-cyber/rs_grounding_project/releases/download/11.26/vehicle_fusion_best_0.553239_20261126.tar.gz.sha256) | 验证下载文件与上传文件一致 |
| [评分协议](scoring_protocol.json) | 各任务查询数、类别占比、公式与模型分工 |
| [压缩包完整性记录](archive_integrity.json) | 原始归档逐成员读取记录 |

大权重作为同仓库 Release 附件保存，本目录提供说明、校验记录和下载入口。

**原始归档的尾部不完整。** 已读取到 `CLIP_DINO_global_best_fusion_export_20261125/fusion.safetensors` 时出现 EOF。前段的车辆 YOLO 权重、排序头、记录，以及 CLIP 和 DINO 主权重可完整读出，但仍需补齐融合权重及缺失配置/校验文件后才能完整恢复。Release 已标为预发布。文件上传后与本地 SHA256 一致，不能据此认为原压缩包已经修复。

```text
SHA256  bc2ebd92bd2880e2d4ce52fce124a68c3823451b73b1deaa0baca168ebc417ed
```

Linux 下载与校验：

```bash
curl -fL -O https://github.com/aalandbartelt-cyber/rs_grounding_project/releases/download/11.26/vehicle_fusion_best_0.553239_20261126.tar.gz
curl -fL -O https://github.com/aalandbartelt-cyber/rs_grounding_project/releases/download/11.26/vehicle_fusion_best_0.553239_20261126.tar.gz.sha256
sha256sum -c vehicle_fusion_best_0.553239_20261126.tar.gz.sha256
```

## 评分备注

总分使用官网公布的公式：

`0.1 × R@1 + 0.2 × R@5 + 0.6 × Acc@0.5 + 0.1 × Acc@0.75`。

检索指标使用检索查询总数作分母，定位指标使用定位查询总数作分母，每条查询等权。四类按实际查询数量占比参与汇总，没有给车辆额外增加评分权重，也没有对四类做各占 25% 的宏平均。

| 类别 | 各任务查询数 | 占比 |
| --- | ---: | ---: |
| 油罐 | 408 | 22.99% |
| 舰船 | 159 | 8.96% |
| 车辆 | 272 | 15.32% |
| 飞机 | 936 | 52.73% |
| 合计 | 1775 | 100% |

实验使用 371 张图图库、各 1775 条检索/定位查询。0.553239 是这一内部场景留出范围下的分数，不是官网排行榜成绩；原 CLIP/DINO 训练时已见过完整公开 train。YOLO 对车辆进行专项检测属于模型分工，不等于评分偏向车辆。

不同样本范围及图库下的总分不能直接比较。主说明中的分数与历史 `11.19` 目录中的成绩属于不同实验范围。
