"""Stream verified code/data delivery archives without loading their contents into RAM."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from common import REPO, Resources, read, write, sha


def archive(files,root,destination):
    if destination.exists(): raise FileExistsError(destination)
    with ZipFile(destination,"x") as zipped:
        for path in files:
            compression=ZIP_STORED if path.suffix.lower() in (".jpg",".jpeg",".png") else ZIP_DEFLATED
            zipped.write(path,path.relative_to(root).as_posix(),compress_type=compression)
    with ZipFile(destination) as zipped:
        bad=zipped.testzip()
        if bad is not None: raise ValueError(f"Archive CRC validation failed: {bad}")
    return {"path":str(destination),"bytes":destination.stat().st_size,"sha256":sha(destination),"files":len(files)}


def build(dataset,out):
    resources=Resources()
    if out==dataset or dataset in out.parents or out==REPO or REPO in out.parents:
        raise ValueError("Delivery output must be outside dataset and repository")
    for filename in ("Z_CPU准备工具_20261123.zip","训练增强集_含标注_20261123.zip","交付校验清单.json","上传与使用说明.md"):
        if (out/filename).exists(): raise FileExistsError("Refusing to overwrite an existing delivery file")
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(dataset/"training_manifest.json")
    for item in manifest["items"]:
        for key,hash_key in (("image_path","image_sha256"),("json_label_path","label_json_sha256"),("yolo_label_path","label_yolo_sha256")):
            path=(dataset/item[key]).resolve()
            if dataset.resolve() not in path.parents or sha(path)!=item[hash_key]: raise ValueError("Dataset integrity check failed before packaging")
    code=[]
    for dirname in ("z_evaluation","z_pipeline","z_research"):
        code += [p for p in sorted((REPO/dirname).rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py",".ps1",".md",".txt",".json")]
    for filename in ("目录导航.md","目录说明.md"):
        if (REPO/filename).exists(): code.append(REPO/filename)
    data=[p for p in sorted(dataset.rglob("*")) if p.is_file()]
    code_entry=archive(code,REPO,out/"Z_CPU准备工具_20261123.zip")
    data_entry=archive(data,dataset,out/"训练增强集_含标注_20261123.zip")
    report={"code":code_entry,"dataset":data_entry,"training_sources":len(manifest["variants_by_source"]),
            "dataset_images":len(manifest["items"]),"dataset_contains_heldout_images":False,
            "dataset_purpose":"detection_training; augmented text requires review",
            "resources":resources.report()}
    write(out/"交付校验清单.json",report)
    (out/"上传与使用说明.md").write_text(
        "# 上传与使用\n\n解压代码包到服务器项目code目录，解压训练增强集到独立datasets目录。\n\n"
        "代码包不含模型权重与私有评测GT；增强集只含冻结346张训练原图的版本，不含开发或验证图片。\n\n"
        "训练增强集保留相对路径，包含COCO、YOLO、逐图JSON。增强文本尚未审核，首轮只用于检测。\n\n"
        "云端仍需另外上传原始训练包并按冻结名单重建评测输入。先运行hpc/probe_runtime.py，再安装与实际镜像匹配的依赖。\n\n"
        "不得按全部增强图片数直接增加训练更新次数。先按原图采样，进行固定预算的增强消融。\n\n"
        "交付校验清单.json包含ZIP大小与SHA256；传输后应核对。当前未执行模型训练或上传GitHub。\n",
        encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    build(args.dataset.resolve(),args.out.resolve())
