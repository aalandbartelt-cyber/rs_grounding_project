"""Export fully labelled detection variants from frozen training images only."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from common import REPO, Resources, fresh_output, read, write, sha, valid_box
from geometry import transform
from preview import OPERATIONS, clip, crop_window, render
from evalkit import load_data, source_inventory, jpeg_size

OP_NAMES={"original":"原图训练副本", "brightness_095":"亮度轻降至95%", "resample_blur_085":"缩小至85%后复原的轻度模糊",
          "jpeg_80":"JPEG质量80压缩", "flip_x":"水平翻转", "rotate90_ccw":"逆时针旋转90度", "crop_keep_all":"保留全部目标的裁剪"}


def yolo_line(category,box,width,height):
    x,y,w,h=box
    values=[(x+w/2)/width,(y+h/2)/height,w/width,h/height]
    if category not in (1,2,3,4) or any(v < -1e-8 or v > 1+1e-8 for v in values):
        raise ValueError("YOLO class/coordinate conversion failed")
    return str(category-1)+" "+" ".join(f"{v:.12f}" for v in values)


def labelme_shape(name,category,box,text,operation,index):
    x,y,w,h=box
    return {"label":name,"category_id":category,"points":[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],
        "shape_type":"rectangle","group_id":None,"description":text if operation=="original" else "",
        "original_description_for_review":text,"source_shape_index":index,
        "flags":{"detection_only":operation!="original","text_review_required":operation!="original"}}


def export(data,output_path,operations):
    resources=Resources()
    data=data.resolve()
    if output_path.resolve()==data or data in output_path.resolve().parents:
        raise ValueError("Do not write into source data")
    manifest_path=REPO/"z_evaluation"/"splits"/"manifest.json"
    manifest=read(manifest_path)
    if source_inventory(data)!=manifest["source_sha256"]:
        raise ValueError("Frozen source fingerprints differ")
    records,_,mapping=load_data(data)
    training=[r for r in records if r["key"] in set(manifest["train_keys"])]
    if {r["key"] for r in training}!=set(manifest["train_keys"]):
        raise ValueError("Training split mismatch")
    if len(set(operations))!=len(operations) or not set(operations)<=set(OPERATIONS):
        raise ValueError("Invalid or repeated augmentation operations")
    output=fresh_output(output_path)
    categories=[{"id":cid,"name":name} for name,cid in sorted(mapping.items(),key=lambda item:item[1])]
    names={cid:name for name,cid in mapping.items()}
    (output/"classes.txt").write_text("\n".join(names[i] for i in range(1,5))+"\n",encoding="utf-8")
    write(output/"类别映射.json",[{"name":names[i],"category_id":i,"yolo_class_id":i-1} for i in range(1,5)])
    coco={op:{"images":[],"annotations":[],"categories":categories,
              "info":{"purpose":"detection_training","operation":op,"frozen_manifest_sha256":sha(manifest_path)}} for op in operations}
    jobs,items=[],[]
    by_source=collections.defaultdict(dict)
    adjustments=[]
    for record in training:
        w,h=record["width"],record["height"]
        boxes=[clip(obj["bbox"],w,h) for obj in record["objects"]]
        window=crop_window(boxes,w,h) if boxes else [0,0,w,h]
        source=data/record["split"]/"images"/(record["stem"]+".jpg")
        for obj,box in zip(record["objects"],boxes):
            if box!=obj["bbox"]:
                adjustments.append({"source_key":record["key"],"shape_index":obj["shape_index"],
                                    "original_bbox":obj["bbox"],"visible_training_bbox":box})
        for op in operations:
            affine,new_w,new_h=transform(op,w,h,window)
            changed=[affine.box(box) for box in boxes]
            for original,box in zip(boxes,changed):
                if not valid_box(box) or min(box[:2]) < -1e-6 or box[0]+box[2]>new_w+1e-6 or box[1]+box[3]>new_h+1e-6:
                    raise ValueError("Invalid augmented training box")
                if any(abs(a-b)>1e-6 for a,b in zip(original,affine.inverse().box(box))):
                    raise ValueError("Training box mapping roundtrip failed")
            stem=record["key"].replace("/","_")+"__"+op
            image_path=Path("images")/op/(stem+".jpg")
            yolo_path=Path("labels")/op/(stem+".txt")
            json_path=Path("labels_json")/op/(stem+".json")
            jobs.append({"source":str(source),"destination":str(output/image_path),"operation":op,"overlay":None,
                         "boxes":[],"crop":window,"width":new_w,"height":new_h})
            yolo_file=output/yolo_path
            yolo_file.parent.mkdir(parents=True,exist_ok=True)
            lines=[yolo_line(obj["category_id"],box,new_w,new_h) for obj,box in zip(record["objects"],changed)]
            yolo_file.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
            shape_labels=[labelme_shape(names[obj["category_id"]],obj["category_id"],box,obj["text"],op,obj["shape_index"])
                          for obj,box in zip(record["objects"],changed)]
            write(output/json_path,{"version":"5.0.0","flags":{"augmented":op!="original","detection_training":True},
                "shapes":shape_labels,"imagePath":f"../../images/{op}/{stem}.jpg","imageData":None,
                "imageWidth":new_w,"imageHeight":new_h,
                "augmentation":{"source_key":record["key"],"operation":op,"affine":affine.values(),
                    "inverse_affine":affine.inverse().values(),"joint_text_training_allowed":op=="original",
                    "text_review":"original_text" if op=="original" else "pending_do_not_train_text"}})
            image_id=len(coco[op]["images"])+1
            coco[op]["images"].append({"id":image_id,"file_name":image_path.as_posix(),"width":new_w,"height":new_h,
                                       "source_key":record["key"]})
            for obj,box in zip(record["objects"],changed):
                coco[op]["annotations"].append({"id":len(coco[op]["annotations"])+1,"image_id":image_id,
                    "category_id":obj["category_id"],"bbox":box,"area":box[2]*box[3],"iscrowd":0})
            item={"source_key":record["key"],"source_image_sha256":sha(source),"operation":op,"operation_zh":OP_NAMES[op],
                  "image_path":image_path.as_posix(),"yolo_label_path":yolo_path.as_posix(),"json_label_path":json_path.as_posix(),
                  "width":new_w,"height":new_h,"objects":len(changed),"affine":affine.values(),
                  "inverse_affine":affine.inverse().values(),"eligible_for_detection_training":True,
                  "eligible_for_joint_text_training":op=="original","text_review_required":op!="original",
                  "crop_window":window if op=="crop_keep_all" else None,
                  "crop_is_full_image":op=="crop_keep_all" and window==[0,0,w,h]}
            by_source[record["key"]][op]=len(items)
            items.append(item)
        resources.check()
    render(jobs,output)
    for item in items:
        image=output/item["image_path"]
        if jpeg_size(image)!=(item["width"],item["height"]):
            raise ValueError("Image and label dimensions disagree")
        item["image_sha256"]=sha(image)
        item["label_json_sha256"]=sha(output/item["json_label_path"])
        item["label_yolo_sha256"]=sha(output/item["yolo_label_path"])
    for op,annotation in coco.items():
        write(output/"annotations"/(op+".coco.json"),annotation)
    write(output/"training_manifest.json",{"schema_version":"1.0","source_split":"frozen_train_only",
        "source_manifest_sha256":sha(manifest_path),"items":items,"variants_by_source":dict(by_source),
        "category_ids":{"coco_and_labelme":"1油罐,2舰船,3车辆,4飞机","yolo":"0油罐,1舰船,2车辆,3飞机"}})
    write(output/"training_recipe.template.json",{"status":"configuration_only_no_training_performed",
        "source_images_per_epoch":len(training),"sample_unit":"original_source_key",
        "variant_sampling":"Choose one allowed variant per sampled source; keep optimizer updates comparable",
        "baseline_operations":["original"],"candidate_operations":list(operations),
        "mandatory_original_probability_minimum_initial_hypothesis":0.5,
        "joint_text_operations":["original"],"seed":20261123,
        "notes":"This probability is an initial hypothesis, not a validated best setting; do not concatenate all variants then compare equal epoch counts"})
    report={"source_images":len(training),"source_objects":sum(len(r["objects"]) for r in training),
        "exported_images":len(items),"exported_objects":sum(item["objects"] for item in items),
        "operations":list(operations),"per_operation":{op:{"images":len(v["images"]),"objects":len(v["annotations"])} for op,v in coco.items()},
        "training_box_adjustments":adjustments,"all_geometry_checks_passed":True,
        "all_saved_dimensions_passed":True,"all_sources_in_train":True,
        "source_unchanged":source_inventory(data)==manifest["source_sha256"],
        "augmented_text_labels":"withheld_pending_semantic_review; original_description_for_review is not a training label",
        "resources":resources.report(),"renderer_resources":read(output/"render_resources.json")}
    write(output/"export_report.json",report)
    (output/"使用说明.md").write_text(
        "# 带标注的检测训练增强集\n\n"
        f"固定训练原图{len(training)}张，导出{len(items)}张图片。仅原始训练划分，不包含开发/验证图片。\n\n"
        "- images/<操作>/：模型输入图片。labels/<操作>/：对应YOLO标签，类别为0至3。\n"
        "- labels_json/<操作>/：逐图矩形与来源说明；COCO汇总位于annotations/，类别为1至4。\n"
        "- training_manifest.json：原图来源、变换、图片与标签相对路径及SHA256。所有路径均相对此目录，上传Linux后可使用。\n"
        "- 增强图的description字段暂留空，原文放在original_description_for_review，避免误当成正确的增强文本。\n"
        "- 原图副本可用于原有图文监督；其余版本首轮只用于检测。若要用于图文训练，需完成文本与参照物审核。\n"
        "- 同一原图的多个版本不是独立场景。公平对照按原图采样、固定训练更新次数，不因样本数变多就多训练七倍。\n"
        "- COCO检测框从训练副本的可见边界变换，原始JSON与评测GT不变；类别1至4映射到模型内部0至3应由接入层明确处理。\n"
        "- 保框裁剪保留全部标注目标，但不保证保留文字中的未标注参照物；不作为中文定位的默认裁剪方案。\n",
        encoding="utf-8")
    print({k:report[k] for k in ("source_images","source_objects","exported_images","exported_objects","resources","renderer_resources")},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--operations",nargs="+",choices=OPERATIONS,default=list(OPERATIONS))
    args=parser.parse_args()
    export(args.data,args.out,args.operations)
