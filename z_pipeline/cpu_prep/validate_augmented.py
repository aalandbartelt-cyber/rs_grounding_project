"""Cross-check COCO, YOLO and Labelme exports against frozen source annotations."""
import argparse
import collections
import math
from pathlib import Path

from common import REPO, Resources, read, write, sha
from evalkit import load_data, source_inventory, bbox_from_points, jpeg_size


def near(a,b,tolerance=1e-6):
    return len(a)==len(b) and all(abs(x-y)<=tolerance for x,y in zip(a,b))


def validate(data,dataset,out):
    resources=Resources()
    if out.exists():
        raise FileExistsError("Choose a new validation report filename")
    manifest=read(REPO/"z_evaluation"/"splits"/"manifest.json")
    exported=read(dataset/"training_manifest.json")
    records,_,_=load_data(data)
    records={r["key"]:r for r in records}
    if exported["source_manifest_sha256"]!=sha(REPO/"z_evaluation"/"splits"/"manifest.json"):
        raise ValueError("Wrong frozen manifest")
    train=set(manifest["train_keys"])
    if set(exported["variants_by_source"])!=train:
        raise ValueError("Augmented dataset does not cover exactly the frozen training images")
    operations={item["operation"] for item in exported["items"]}
    coco={op:read(dataset/"annotations"/(op+".coco.json")) for op in operations}
    coco_images={op:{r["file_name"]:r for r in d["images"]} for op,d in coco.items()}
    coco_objects={}
    for op,document in coco.items():
        if {r["id"] for r in document["categories"]}!={1,2,3,4}:
            raise ValueError("COCO categories do not match competition IDs")
        if len({r["id"] for r in document["annotations"]})!=len(document["annotations"]):
            raise ValueError("Repeated COCO annotation IDs")
        grouped=collections.defaultdict(list)
        for annotation in document["annotations"]:
            grouped[annotation["image_id"]].append(annotation)
        coco_objects[op]=grouped
    count,objects=0,0
    for item in exported["items"]:
        source_key=item["source_key"]
        if source_key not in train:
            raise ValueError("Held-out source appeared in training export")
        op=item["operation"]
        for file_key,hash_key in (("image_path","image_sha256"),("json_label_path","label_json_sha256"),("yolo_label_path","label_yolo_sha256")):
            path=(dataset/item[file_key]).resolve()
            if dataset.resolve() not in path.parents or sha(path)!=item[hash_key]:
                raise ValueError("Missing, escaping or modified dataset file")
        labels=read(dataset/item["json_label_path"])
        width,height=jpeg_size(dataset/item["image_path"])
        if (width,height)!=(labels["imageWidth"],labels["imageHeight"]) or (width,height)!=(item["width"],item["height"]):
            raise ValueError("Image dimensions differ from label dimensions")
        relative_image=((dataset/item["json_label_path"]).parent/labels["imagePath"]).resolve()
        if relative_image!=(dataset/item["image_path"]).resolve():
            raise ValueError("Labelme imagePath does not resolve to corresponding image")
        yolo=[line.split() for line in (dataset/item["yolo_label_path"]).read_text().splitlines() if line.strip()]
        source=records[source_key]
        coco_image=coco_images[op][item["image_path"]]
        annotations=coco_objects[op][coco_image["id"]]
        if not (len(yolo)==len(labels["shapes"])==len(source["objects"])==len(annotations)==item["objects"]):
            raise ValueError("Object counts disagree between formats")
        for shape,row,annotation,original in zip(labels["shapes"],yolo,annotations,source["objects"]):
            if len(row)!=5 or int(row[0])+1!=shape["category_id"] or shape["category_id"]!=original["category_id"]:
                raise ValueError("YOLO/JSON class mapping mismatch")
            values=list(map(float,row[1:]))
            if not all(math.isfinite(v) and 0<=v<=1 for v in values):
                raise ValueError("Invalid normalized YOLO coordinates")
            cx,cy,bw,bh=values
            yolo_box=[(cx-bw/2)*width,(cy-bh/2)*height,bw*width,bh*height]
            json_box=bbox_from_points(shape["points"])
            if not near(yolo_box,json_box) or not near(annotation["bbox"],json_box) or annotation["category_id"]!=shape["category_id"]:
                raise ValueError("Coordinate agreement failed across export formats")
            if op!="original" and (shape["description"] or not shape["flags"]["text_review_required"]):
                raise ValueError("Unreviewed augmented description became a training label")
            if shape["original_description_for_review"]!=original["text"]:
                raise ValueError("Source description provenance lost")
            if op=="original" and shape["description"]!=original["text"]:
                raise ValueError("Original text label changed")
            objects+=1
        count+=1
        resources.check()
    inventory=source_inventory(data)
    if inventory!=manifest["source_sha256"]:
        raise ValueError("Original source inventory changed")
    report={"passed":True,"images":count,"objects":objects,"operations":sorted(operations),
        "verified_source_files":len(inventory),"heldout_images_in_training":0,
        "formats_checked":["COCO","YOLO","Labelme JSON"],"resources":resources.report()}
    write(out,report)
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    validate(args.data.resolve(),args.dataset.resolve(),args.out)
