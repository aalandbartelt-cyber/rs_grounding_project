"""Qwen retrieval scoring and candidate selection on original-training inputs."""
import json
import math
import re

import torch
from PIL import Image, ImageDraw

from box_parser import parse
from common import ROOT, TrainArchive, crop, load_training_cache, write_json
from vlm import Qwen


def panel(image, boxes):
    """Show only predicted candidates, with context crops and an overview."""
    overview = image.copy()
    overview.thumbnail((512, 512))
    draw = ImageDraw.Draw(overview)
    sx, sy = overview.width / image.width, overview.height / image.height
    for number, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box
        rect = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        draw.rectangle(rect, outline="red", width=2)
        draw.text((rect[0], max(0, rect[1] - 12)), str(number), fill="red",
                  stroke_width=1, stroke_fill="white")
    columns = min(3, max(1, len(boxes)))
    rows = math.ceil(len(boxes) / columns)
    canvas = Image.new("RGB", (overview.width + columns * 180, max(overview.height, rows * 200)), "white")
    canvas.paste(overview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes):
        view = crop(image, box, 1.3)
        view.thumbnail((172, 172))
        x, y = overview.width + index % columns * 180, index // columns * 200
        canvas.paste(view, (x, y + 20))
        draw.text((x + 2, y + 2), str(index + 1), fill="black")
    return canvas


def main():
    torch.set_num_threads(4)
    cache = load_training_cache()
    jobs = json.loads((ROOT / "features/jobs.json").read_text(encoding="utf-8"))
    geoground = json.loads((ROOT / "features/geoground_answers.json").read_text(encoding="utf-8"))
    units = json.loads((ROOT / "results/geoground_coordinate_units.json").read_text())["units"]
    retrieval_path, grounding_path = ROOT / "features/qwen_retrieval.json", ROOT / "features/qwen_grounding.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else {}
    grounding = json.loads(grounding_path.read_text(encoding="utf-8")) if grounding_path.exists() else {}
    model, data = Qwen(), TrainArchive()
    for number, job in enumerate(jobs["retrieval"], 1):
        key = f"{job['query']}:{job['image']}"
        if key not in retrieval:
            image = data.image(job["label"])
            boxes = cache["records"][job["image"]]["boxes"]
            view = crop(image, boxes[job["candidate"]], 1.5) if len(boxes) else image
            prompt = ("Does the aircraft in this image match the following description? "
                      "Consider visible color, engine count and arrangement, and surroundings. "
                      "Answer only Yes or No.\nDescription: " + job["text"])
            retrieval[key] = {"yes_minus_no_logit": model.match_score(prompt, view),
                              "candidate": job["candidate"], "input_kind": "predicted region crop"}
            write_json(retrieval_path, retrieval)
        if number % 10 == 0 or number == len(jobs["retrieval"]):
            print(f"Qwen retrieval {number}/{len(jobs['retrieval'])}", flush=True)
    for number, job in enumerate(jobs["grounding"], 1):
        key = f"{job['query']}:{job['image']}"
        if key not in grounding:
            row = cache["records"][job["image"]]
            image = data.image(job["label"])
            boxes = row["boxes"].tolist()
            answer = geoground[key]["answer"]
            geo_box = parse(answer, image.width, image.height, units)
            if geo_box is not None:
                boxes.append(geo_box.tolist())
            if boxes:
                prompt = (f"左侧是遥感图像和候选框编号，右侧是各候选的局部放大图。"
                          f"根据下列描述，从编号1到{len(boxes)}中选择最符合描述的目标。"
                          "只输出一个编号，无法判断则输出0。描述：" + job["text"])
                answer = model.answer(prompt, panel(image, boxes), max_tokens=16)
                digits = re.findall(r"\d+", answer)
                chosen = int(digits[0]) - 1 if digits and 0 < int(digits[0]) <= len(boxes) else -1
            else:
                answer, chosen = "", -1
            grounding[key] = {"answer": answer, "chosen": chosen, "boxes": boxes,
                              "dino_candidates": len(row["boxes"]), "geoground_box_added": geo_box is not None}
            write_json(grounding_path, grounding)
        if number % 10 == 0 or number == len(jobs["grounding"]):
            print(f"Qwen grounding {number}/{len(jobs['grounding'])}", flush=True)
    data.audit(ROOT / "results/Qwen_data_access.json")
    data.close()


if __name__ == "__main__":
    main()
