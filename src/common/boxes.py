"""框几何：IoU、坐标转换、裁剪、面积。"""


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def clip_xyxy(box, width, height):
    x1 = min(max(0.0, float(box[0])), float(width))
    y1 = min(max(0.0, float(box[1])), float(height))
    x2 = min(max(0.0, float(box[2])), float(width))
    y2 = min(max(0.0, float(box[3])), float(height))
    return [x1, y1, x2, y2]
