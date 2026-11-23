"""Write a concise Chinese record of the completed training-only fusion round."""
import json
from pathlib import Path

from common import ROOT, write_json


def main():
    report = json.loads((ROOT / "results/score_round.json").read_text(encoding="utf-8"))
    names = {
        "ChineseCLIP_DINO_frozen": "冻结 Chinese-CLIP + GroundingDINO 基线",
        "RemoteCLIP_DINO": "RemoteCLIP + GroundingDINO",
        "GeoRSCLIP_DINO": "GeoRSCLIP + GroundingDINO",
        "equal_remote_fusion": "两套遥感 CLIP 等权融合 + GroundingDINO",
        "four_frameworks_equal": "遥感检索 + GeoGround + Qwen3-VL 固定权重",
        "fit_selected_fusion": "仅用拟合图选择的融合权重"}
    lines = [
        "# 遥感模型融合与训练内部评分",
        "",
        f"本轮使用公开训练包中已复核的 49 张民航图。按图像重叠组划分为 {report['fit_images']} 张拟合图"
        f"和 {report['dev_images']} 张内部验证图。下表给出这一次内部验证的结果。",
        "",
        "| 方案 | R@1 | R@5 | Acc@0.5 | Acc@0.75 | 总分，满分 1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in report["results"].items():
        values = [f"{row[metric]*100:.2f}%" for metric in ("R@1", "R@5", "Acc@0.5", "Acc@0.75")]
        lines.append("| " + names[key] + " | " + " | ".join(values) + f" | {row['score_out_of_1']:.4f} |")
    chosen = report["results"]["fit_selected_fusion"]
    lines.extend([
        "",
        "总分按 0.1 × R@1 + 0.2 × R@5 + 0.6 × Acc@0.5 + 0.1 × Acc@0.75 计算。"
        "参数在拟合图上选择，并在内部验证评分前写入 frozen_config.json。"
        "这次内部验证的图库与早先 47 张留出图不同，图库大小和场景构成会影响分数。",
        "",
        "RemoteCLIP 和 GeoRSCLIP 对整图、GroundingDINO 候选及其上下文编码。"
        "Qwen3-VL 将中文查询翻译成英文供遥感编码器使用，并对检索前五候选打分。"
        "GeoGround 根据查询产生区域框，Qwen3-VL 再结合整图和局部放大图选择候选。"
        "拟合阶段校准遥感分数比例、检测置信度、GeoGround 区域支持和 Qwen 匹配分数。",
        "本轮采用分数级融合：各网络的预训练参数保持冻结，只在拟合图上选择融合系数。"
        "未执行网络微调，也未生成新的网络适配器权重。融合系数另存为 weights/fusion_coefficients.json。",
        "",
        "检索重排限制在预先召回的前五张图，因此这部分重排主要影响 R@1。"
        "GeoGround 与 Qwen3-VL 采用 NF4 4 位量化，在本机 RTX 4060 Laptop GPU 上运行。"
        "其量化结果需要与全精度模型区别记录。",
        "",
        "权重位于本目录 weights 中。RemoteCLIP、GeoRSCLIP、GeoGround 与 Qwen3-VL 来自各自官方权重仓库，"
        "下载清单记录固定版本、文件大小和 SHA-256。GroundingDINO 使用此前已校验的官方预训练权重副本。",
        "",
        "| 框架 | 用途 | 官方权重来源 |",
        "|---|---|---|",
        "| RemoteCLIP ViT-L/14 | 遥感图文检索与区域相似度 | https://huggingface.co/chendelong/RemoteCLIP |",
        "| GeoRSCLIP ViT-L/14-336 | 第二路遥感图文特征 | https://huggingface.co/Zilun/GeoRSCLIP |",
        "| GeoGround LLaVA-7B | 文本指向的区域框 | https://huggingface.co/erenzhou/GeoGround |",
        "| Qwen3-VL-4B-Instruct | 查询翻译、候选匹配与选择 | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct |",
        "| GroundingDINO tiny | 复用冻结检测器产生的候选 | https://huggingface.co/IDEA-Research/grounding-dino-tiny |",
        "",
        "Chinese-CLIP 对照组仅复用原始训练图的冻结预训练特征；未加载旧适配器。"
        "本轮实际硬件和依赖版本见 results/environment.json。",
        "",
        "data/original_train_only.zip 仅包含 49 张原始 train 图像及对应标注。"
        "读取接口只允许访问这些条目。SIFT 检查将有重叠的图像保持在同一分组，"
        "既往用全部 49 张图训练过的适配器未接入本轮。模型推理进程通过审计钩子禁止网络连接与域名解析。",
        "",
        "逐项指标和命中数见 [score_round.json](results/score_round.json)，"
        "划分与重叠分组见 [split.json](results/split.json)，"
        "最终参数见 [frozen_config.json](results/frozen_config.json)，"
        "权重来源见 [download_manifest.json](weights/download_manifest.json)。",
        "GroundingDINO 副本校验见 weights/GroundingDINO_provenance.json，"
        "最终文件、数据访问记录与分数验算见 results/final_verification.json。",
        "",
        "已下载权重并提取两套遥感视觉特征后，可在 PowerShell 中运行以下命令复现本轮后续步骤。",
        "",
        "```powershell",
        "& ./code/run_round.ps1",
        "```",
    ])
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded one original-training dev round: {chosen['score_out_of_1']:.4f}/1", flush=True)


if __name__ == "__main__":
    main()
