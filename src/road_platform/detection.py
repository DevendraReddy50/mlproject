"""Optional Ultralytics YOLO inference. Supply licensed weights via YOLO_WEIGHTS."""
import logging
from pathlib import Path
import cv2
log=logging.getLogger(__name__)
class DamageDetector:
    def __init__(self,weights:str|None):
        if not weights: raise RuntimeError('YOLO_WEIGHTS is unset; detection requires a trained/authorized damage-detection weight file.')
        if not Path(weights).is_file(): raise FileNotFoundError(f'YOLO weights not found: {weights}')
        try:
            from ultralytics import YOLO
            self.model=YOLO(weights)
        except Exception as e: raise RuntimeError(f'Unable to load YOLO model: {e}') from e
    def predict(self,image_path:str,confidence=.25):
        image=cv2.imread(image_path)
        if image is None: raise ValueError(f'Cannot read image: {image_path}')
        result=self.model.predict(image,conf=confidence,verbose=False)[0]
        return [{'class_id':int(b.cls.item()),'label':self.model.names[int(b.cls.item())],'confidence':float(b.conf.item()),'box_xyxy':[float(x) for x in b.xyxy[0].tolist()]} for b in result.boxes]

def train_yolo(data_yaml:str,weights='yolov8n.pt',epochs=50,imgsz=640,project='models/yolo'):
    """Train YOLO from a user-provided, correctly annotated Ultralytics dataset YAML."""
    if not Path(data_yaml).is_file(): raise FileNotFoundError(f'Dataset YAML not found: {data_yaml}')
    try:
        from ultralytics import YOLO
        model=YOLO(weights)
        return model.train(data=data_yaml,epochs=epochs,imgsz=imgsz,project=project,name='road_damage')
    except Exception as e:
        log.exception('YOLO training failed')
        raise RuntimeError(f'YOLO training failed: {e}') from e
