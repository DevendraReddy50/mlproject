import argparse
import json
import logging
from .config import settings
from .data import load_data
from .preprocessing import clean_data
from .eda import summarize,save_eda
from .severity import train_severity
from .training import train_models
from .logging_config import configure_logging

def main():
    configure_logging(); p=argparse.ArgumentParser(); p.add_argument('command',choices=['data','eda','train','all','train-yolo']); p.add_argument('--real-only',action='store_true'); p.add_argument('--yolo-data',default='data/yolo/data.yaml'); p.add_argument('--yolo-weights',default='yolov8n.pt'); a=p.parse_args()
    if a.command=='train-yolo':
        from .detection import train_yolo
        train_yolo(a.yolo_data,a.yolo_weights); return
    df=clean_data(load_data(settings.data_path,allow_synthetic=not a.real_only))
    if a.command in ('data','all'): logging.info('Rows: %d',len(df))
    if a.command in ('eda','all'): print(json.dumps(summarize(df),indent=2,default=str)); save_eda(df)
    if a.command in ('train','all'):
        print(json.dumps({'severity':train_severity(df,settings.model_dir),'forecast':train_models(df,settings.model_dir)},indent=2))
if __name__=='__main__':main()


