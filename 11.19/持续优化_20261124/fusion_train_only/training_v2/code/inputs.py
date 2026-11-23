"""Shared, original-train-only inputs for the second optimization round."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'training_v2'
sys.path.insert(0,str(BASE/'training_v1/code'))
sys.path.insert(0,str(BASE/'code'))

from data import Data
from common import TrainArchive,crop,digest,write_json
from box_parser import parse,iou


def inverse_box(box,width,height,rotation):
    """Map a PIL counter-clockwise image rotation's XYXY box back to the source."""
    import torch
    x1,y1,x2,y2=box.tolist()
    if rotation==0: values=[x1,y1,x2,y2]
    elif rotation==90: values=[width-y2,x1,width-y1,x2]
    elif rotation==180: values=[width-x2,height-y2,width-x1,height-y1]
    elif rotation==270: values=[y1,height-x2,y2,height-x1]
    else: raise ValueError('Unsupported rotation')
    result=torch.tensor(values,dtype=torch.float32)
    result[0::2].clamp_(0,width);result[1::2].clamp_(0,height)
    return result
