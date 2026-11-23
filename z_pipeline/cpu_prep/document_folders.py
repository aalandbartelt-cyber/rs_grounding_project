"""Add Chinese folder notes without touching frozen source/evaluation directories."""
import argparse
from datetime import date
import os
from pathlib import Path

from common import read, write, REPO, sha
from evalkit import source_inventory
from export_augmented import OP_NAMES

RUNTIME={"cache","temp","venv",".venv","__pycache__"}


def purpose(relative):
    p=relative.as_posix()
    name=relative.name
    if name=="__pycache__": return "Python自动生成的字节码缓存，可由程序重建，不包含原始数据或训练标注。"
    if p==".": return "图像识别项目工作区：原始数据、Z分支代码、评测与实验产物均在此。"
    if p.startswith("rs_grounding_train_v1"):
        if name=="images": return "官方原始JPEG图片。只读保留；训练/开发用途以冻结名单决定。"
        if name=="labels_json": return "官方逐图JSON：目标类别、中文描述和浮点矩形框，作为主要标注来源。只读保留。"
        if name=="labels_xml": return "官方VOC XML标注副本，部分坐标与JSON存在取整差异；不替代文本配对的JSON真值。"
        if name=="train": return "官方原train共376张；其中30张已冻结留作开发，当前训练只用其余346张。"
        if name=="val": return "官方原val共51张，作为补充阶段验证，未加入训练。"
        return "官方原始训练包，1284个文件指纹冻结；不在此添加说明文件，作用统一记录在工作区总览。"
    if p.startswith("_eval_transfer_"): return "历史评测传输文件归档，便于追溯；当前实现以Z分支z_evaluation和冻结v2输出为准。"
    if p.startswith("_previous_eval"): return "历史评测预留/归档目录；当前不作为训练数据或模型输出入口。"
    if p.startswith("rs_grounding_project_Z"):
        q="/".join(relative.parts[1:])
        exact={"":"GitHub项目本地Z分支。保留main原有内容，Z的新工作位于z_evaluation、z_pipeline、z_research。",
            "11.19":"仓库已有11.19民航客机方案归档，包含方法、结果与提交工程，仅作参考。",
            "11.19/method":"11.19历史方案的方法代码与说明。",
            "11.19/results":"11.19历史实验与平台结果，不能混作当前v2协议的新成绩。",
            "11.19/submission":"11.19历史Docker提交工程。",
            "A100环境调试":"仓库已有A100环境调试过程及首轮结果，不代表当前云账号已配置完成。",
            "A100环境调试/code":"历史环境准备、训练与核查代码，以及模型来源清单。",
            "A100环境调试/results":"历史A100调试结果与实验记录。",
            "data":"仓库数据入口，主要保存历史划分；官方图片位于仓库外的原始训练包。",
            "data/splits":"main原方案的历史划分，和Z的v2名单不同；当前实验使用z_evaluation/splits。",
            "docs":"仓库已有赛题审核、分工、思路和开发说明，作为参考资料保留。",
            "reports":"仓库已有分析报告与历史审核产物。",
            "src":"仓库原有数据分析、转换与评测代码，保留以供对照。",
            "src/common":"原方案共用的类别与框坐标工具。",
            "src/detect":"原方案检测数据转换代码。",
            "src/eval":"原方案评测与测试代码；当前Z v2主评测位于z_evaluation。",
            "submission":"原方案提交工程入口，当前没有在此启动新模型部署。",
            "z_evaluation":"Z已上传的可信评测v2：评分器、测试、冻结划分和审核报告。",
            "z_evaluation/audit":"v2评测构建、自检和图片相似性审核摘要。",
            "z_evaluation/docs":"官方训练包说明与本地规则副本，便于核对接口。",
            "z_evaluation/splits":"当前冻结v2训练/开发/验证名单与源文件SHA256。",
            "z_pipeline":"Z独立执行脚本及阶段计划，新增工作不覆盖原方案。",
            "z_pipeline/cpu_prep":"无需GPU的数据审核、增强标注、预测接口、错误诊断、比较和低内存验证工具。",
            "z_pipeline/hpc":"自强5000操作指南与远程环境探测脚本；不包含账号凭证。",
            "z_research":"独立模型研究、选型依据和待验证的实验路线。"}
        if q in exact: return exact[q]
    if p.startswith("Z_model_lab"):
        if name in RUNTIME: return {"venv":"本机独立Python环境；不能直接复制成Linux云端环境。","cache":"下载与依赖缓存；不作为训练数据或实验结果。","temp":"本机临时文件；不保存唯一模型检查点。"}.get(name,"运行环境目录，不向内部注入项目说明。")
        if "augmented_detection_v1" in p:
            if name=="augmented_detection_v1": return "仅由346张冻结训练原图导出的带标注检测增强集，含来源、变换和三种标注格式。"
            if name=="images": return "未画框的训练图片，按增强操作分目录。"
            if name=="labels": return "YOLO文本标注根目录：类别0油罐、1舰船、2车辆、3飞机，坐标归一化。"
            if name=="labels_json": return "逐图Labelme JSON根目录：浮点框、类别和来源；增强中文描述留空并保存待审核原文。"
            if name=="annotations": return "按增强操作汇总的COCO检测标注，category_id为1至4。"
            if name in OP_NAMES:
                parent=relative.parent.name
                role={"images":"训练图片","labels":"对应YOLO标签","labels_json":"对应逐图JSON标注"}.get(parent,"数据")
                return f"{OP_NAMES[name]}的{role}。同一原图的版本仍属于同一个训练来源，不作为独立场景。"
        if "data_v2/evaluation_only" in p: return "开发或补充验证的标注副本，仅供独立评测，不得加入训练。"
        if name=="data_v2": return "原始数据的本机训练/评测索引导出；含Windows绝对路径，迁到Linux后重新生成。"
        if p.endswith("data_v2/train"): return "固定346张训练图的COCO框与1913条图文对，不包含开发/验证目标。"
        if "robustness_pilot" in p:
            if name=="images": return "仅用于鲁棒性诊断的退化图库，不能当作增强训练图片。"
            if name=="input_path": return "单个退化条件的独立81图评测输入，保留原查询与ID；不与其他版本合并图库。"
            if name in OP_NAMES: return f"{OP_NAMES[name]}的辅助评测条件，使用原图坐标，不替代主评分。"
            return "固定的明暗、压缩与轻模糊评测副本；只作辅助诊断，尚无真实模型成绩。"
        if name=="previews": return "少量训练增强审核样例及index.html，供检查框与文字含义，不等于完整训练集。"
        if name=="overlays": return "画有绿色标注框的审核图片，严禁作为模型训练输入。"
        if p.endswith("previews/images"): return "未画框的少量增强预览图片；完整训练导出位于augmented_detection_v1。"
        if name=="audit": return "数据统计和训练描述审核队列；关键词标记尚未经过人工语义确认。"
        if name=="contract_fixture": return "测试夹具预测与候选框，仅检验接口；不是模型成绩。"
        if name.startswith("diagnostics_fixture"): return "测试夹具的错误诊断报告；用于核对工具，不代表真实视觉能力。"
        if name.startswith("comparison"): return "配对比较的验证输出；需同一GT、评分器与输入指纹才能比较模型。"
        if name=="cpu_prep_20261123": return "2026-11-23无需GPU的准备阶段产物：审核、增强、标注、接口和诊断。"
        if name.startswith("cpu_"): return "CPU准备流程的独立复现验收目录；与原始输出分开，内部夹具不代表模型成绩。"
        if name=="Z_model_lab": return "仓库外的环境、私有数据索引、增强图片与运行结果；大文件不上传GitHub。"
    if p.startswith("rs_grounding_eval") or p.startswith("Z_eval_outputs"):
        if name=="private": return "评测私有真值，只能由评分/诊断工具读取，模型推理不可读取。"
        if name=="images": return "冻结评测图库原图，未增强，不加入训练。"
        if name=="input_path": return "模型可读评测输入：images、info.json、queries.json，不含真值。"
        if name=="splits": return "冻结图像划分与源文件指纹。"
        if name=="audit": return "评测数据审核和构建摘要。"
        if name=="baselines": return "oracle/全错等评分自检夹具，不是模型性能。"
        if name=="runs" or name=="blind_random": return "历史盲随机接口验证输出，不是经过训练的模型。"
        if name=="docs": return "评测规则及官方数据说明副本。"
        if name=="protocols": return "按协议划分的独立评测输入。"
        if name=="dev4_81": return "四类开发协议：30张查询图、612检索与612定位查询，共用81图图库。"
        if name=="official_val51_81": return "补充验证协议：51张查询图、191检索与191定位查询，共用81图图库。"
        if name=="rs_grounding_eval": return "原独立评测目录，作为历史来源保留；当前代码以Z分支z_evaluation为准。"
        if name=="v2": return "已重建并验证的冻结v2评测输出，继续只读保留。"
        return "本地可信评测产物根目录，固定协议输出位于v2。"
    return f"项目业务子目录“{name}”：具体文件用途由其上级目录说明和工作区总览共同解释。"


def document(root):
    root=root.resolve()
    frozen=[root/"rs_grounding_train_v1",root/"rs_grounding_eval",root/"Z_eval_outputs"/"v2"]
    before=source_inventory(frozen[0])
    rows=[]
    for current,dirs,files in os.walk(root):
        folder=Path(current)
        if root!=folder and root not in folder.resolve().parents:
            raise ValueError("Directory scan escaped project workspace")
        dirs[:]=sorted(name for name in dirs if not name.startswith(".") and not (folder/name).is_symlink())
        relative=folder.relative_to(root)
        protected=any(folder==base or base in folder.parents for base in frozen)
        runtime=folder.name in RUNTIME
        description=purpose(relative)
        status="总览中说明，目录本身不写入" if protected or runtime else "已添加目录说明"
        if runtime:
            dirs[:]=[]
        if not protected and not runtime:
            note=folder/"目录说明.md"
            if not note.exists():
                children="\n".join(f"- `{name}/`：{purpose((folder/name).relative_to(root))}" for name in dirs)
                body=f"# 目录用途\n\n{description}\n\n相对工作区路径：`{relative.as_posix()}`。\n"
                if children: body+="\n## 子目录\n\n"+children+"\n"
                body+="\n说明文件仅作导航。原始数据、冻结划分与历史文件均未改写；本目录内训练/评测用途以对应清单为准。\n"
                note.write_text(body,encoding="utf-8")
            else:
                status="已有说明，保留原内容"
        rows.append({"path":relative.as_posix(),"purpose":description,"status":status})
    index=root/"文件夹用途总览_20261123.md"
    if index.exists():
        index=root/"文件夹用途总览_20261123_补充.md"
    if index.exists(): raise FileExistsError("Folder index already exists; review before regenerating")
    lines=["# 图像识别项目文件夹用途总览","","原始训练包和冻结评测目录不写入说明，以保持文件指纹不变；其每层目录都在本表中说明。运行环境和缓存只备注根目录，不向第三方包内部添加文件。","",
           "| 相对D盘工作区路径 | 中文用途 | 备注方式 |","|---|---|---|"]
    lines += [f"| `{r['path']}` | {r['purpose']} | {r['status']} |" for r in rows]
    index.write_text("\n".join(lines)+"\n",encoding="utf-8")
    repo_index=REPO/"目录导航.md"
    if not repo_index.exists():
        subset=[r for r in rows if r["path"].startswith("rs_grounding_project_Z")]
        repo_index.write_text("# Z分支目录导航\n\n完整工作区说明位于仓库外的 `D:\\计算机视觉\\文件夹用途总览_20261123.md`。大数据和私有结果位于 `Z_model_lab/`，不进GitHub。\n\n"
            +"\n".join(f"- `{r['path'].removeprefix('rs_grounding_project_Z').lstrip('/') or '.'}`：{r['purpose']}" for r in subset)+"\n",encoding="utf-8")
    after=source_inventory(frozen[0])
    if before!=after: raise ValueError("Original training package changed during documentation")
    report={"directories_documented":len(rows),"in_folder_notes":sum(r["status"]=="已添加目录说明" for r in rows),
            "index_only":sum(r["status"]=="总览中说明，目录本身不写入" for r in rows),
            "source_files_unchanged":True,"source_files_verified":len(after),"index":str(index),"details":rows}
    report_path=root/"Z_model_lab"/"cpu_prep_20261123"/"directory_notes_report.json"
    if report_path.exists(): report_path=report_path.with_name("directory_notes_report_补充.json")
    if report_path.exists(): raise FileExistsError("Directory report already exists")
    write(report_path,report)
    print({k:v for k,v in report.items() if k!="details"},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    document(args.root)
