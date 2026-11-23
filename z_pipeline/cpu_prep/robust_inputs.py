"""Prepare separate, fixed photometric stress inputs; never read training labels or GT."""
import argparse
from pathlib import Path
import shutil

from common import Resources, fresh_output, write, sha, read
from contract import load_public_input
from preview import render
from evalkit import jpeg_size


def build(input_path,output_path):
    resources=Resources()
    input_path=input_path.resolve()
    if output_path.resolve()==input_path or input_path in output_path.resolve().parents:
        raise ValueError("Never write into the original input")
    images,queries=load_public_input(input_path)
    output=fresh_output(output_path)
    conditions=("brightness_095","resample_blur_085","jpeg_80")
    jobs=[]
    for condition in conditions:
        root=output/condition/"input_path"
        root.mkdir(parents=True)
        for filename in ("info.json","queries.json"):
            shutil.copyfile(input_path/filename,root/filename)
        for image in images:
            jobs.append({"source":str(image.path),"destination":str(root/"images"/image.path.name),
                "operation":condition,"overlay":None,"boxes":[],"width":image.width,"height":image.height})
    render(jobs,output)
    evidence=[]
    for condition in conditions:
        root=output/condition/"input_path"
        if sha(root/"info.json")!=sha(input_path/"info.json") or sha(root/"queries.json")!=sha(input_path/"queries.json"):
            raise ValueError("Stress input identity changed")
        for image in images:
            if jpeg_size(root/"images"/image.path.name)!=(image.width,image.height):
                raise ValueError("Stress image dimensions changed")
        evidence.append({"condition":condition,"images":len(images),"queries":len(queries),
                         "images_sha256":{str(r.image_id):sha(root/"images"/r.path.name) for r in images}})
    report={"purpose":"evaluation_only_stress_inputs_not_training_data", "source_input":str(input_path),
            "original_images_sha256":{str(r.image_id):sha(r.path) for r in images},
            "conditions":evidence,"query_and_gallery_ids_unchanged":True,
            "curation_status":"pilot_pending_attribute_perceptibility_review",
            "limitations":["These are independent same-size galleries; do not concatenate their variants",
                           "GT boxes remain in the original coordinate frame; no geometric corruption is applied",
                           "Stress conditions do not replace clean primary scoring or certify real-world robustness"],
            "resources":resources.report(),"renderer_resources":read(output/"render_resources.json")}
    write(output/"stress_manifest.json",report)
    print({"conditions":len(conditions),"rendered_images":len(jobs),"resources":report["resources"],
           "renderer_resources":report["renderer_resources"]},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    build(args.input_path,args.out)
