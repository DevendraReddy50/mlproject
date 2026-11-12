import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import settings
from .db import SessionLocal, get_db, healthcheck, init_db
from .models import Prediction, RoadSegment
from .prediction import predict_one
from .logging_config import configure_logging
configure_logging(); log=logging.getLogger(__name__)
app=FastAPI(title='Intelligent Road Condition Assessment API',version='0.1.0')
class SegmentIn(BaseModel):
    id:int; road_name:str; latitude:float=Field(ge=-90,le=90); longitude:float=Field(ge=-180,le=180); pci:float=Field(ge=0,le=100); rainfall_mm:float=Field(ge=0,default=0); traffic_count:int=Field(ge=0,default=0); historical_damage_count:int=Field(ge=0,default=0); pothole_count:int=Field(ge=0,default=0); crack_area_ratio:float=Field(ge=0,le=1,default=0); damage_type:str='none'; severity:str='none'; observed_at:datetime|None=None
@app.on_event('startup')
def startup():
    try:
        init_db(); log.info('Database ready')
        if settings.data_path.exists():
            with SessionLocal() as db:
                if db.query(RoadSegment).count()==0:
                    import pandas as pd
                    frame=pd.read_csv(settings.data_path,parse_dates=['observed_at']).drop_duplicates('segment_id')
                    for _,r in frame.iterrows():
                        db.add(RoadSegment(id=int(r.segment_id),road_name=str(r.get('road_name',f'Road {int(r.segment_id)}')),latitude=float(r.get('latitude',0)),longitude=float(r.get('longitude',0)),pci=float(r.pci),rainfall_mm=float(r.rainfall_mm),traffic_count=int(r.traffic_count),historical_damage_count=int(r.historical_damage_count),pothole_count=int(r.get('pothole_count',0)),crack_area_ratio=float(r.get('crack_area_ratio',0)),damage_type=str(r.get('damage_type','none')),severity=str(r.get('severity','unknown')),observed_at=r.observed_at.to_pydatetime() if hasattr(r.observed_at,'to_pydatetime') else r.observed_at))
                    db.commit(); log.info('Seeded %d segments from configured CSV',len(frame))
    except Exception:log.exception('Database unavailable at startup; health endpoint remains available')
class ForecastIn(BaseModel):
    pci:float=Field(ge=0,le=100); rainfall_mm:float=Field(ge=0,default=0); traffic_count:int=Field(ge=0,default=0); historical_damage_count:int=Field(ge=0,default=0); pothole_count:int=Field(ge=0,default=0); crack_area_ratio:float=Field(ge=0,le=1,default=0); days_since_inspection:int=Field(ge=0,default=30)
@app.get('/health')
def health():
    try:healthcheck(); return {'status':'ok','database':'connected'}
    except Exception as e:log.exception('Health check failed'); raise HTTPException(503,'database unavailable') from e
@app.get('/segments')
def segments(limit:int=Query(500,ge=1,le=5000),db:Session=Depends(get_db)):
    try:return [dict(id=r.id,road_name=r.road_name,latitude=r.latitude,longitude=r.longitude,pci=r.pci,rainfall_mm=r.rainfall_mm,traffic_count=r.traffic_count,historical_damage_count=r.historical_damage_count,pothole_count=r.pothole_count,crack_area_ratio=r.crack_area_ratio,damage_type=r.damage_type,severity=r.severity,observed_at=r.observed_at.isoformat()) for r in db.scalars(select(RoadSegment).order_by(RoadSegment.pci).limit(limit)).all()]
    except Exception as e:log.exception('Segment query failed'); raise HTTPException(503,'database query failed') from e
@app.post('/segments',status_code=201)
def upsert_segment(item:SegmentIn,db:Session=Depends(get_db)):
    try:
        row=db.get(RoadSegment,item.id) or RoadSegment(id=item.id,road_name=item.road_name,latitude=item.latitude,longitude=item.longitude,pci=item.pci,observed_at=item.observed_at or datetime.now(timezone.utc))
        for key,value in item.model_dump(exclude={'id','observed_at'}).items():setattr(row,key,value)
        row.observed_at=item.observed_at or datetime.now(timezone.utc); db.add(row); db.commit(); db.refresh(row); return {'id':row.id,'status':'saved'}
    except Exception as e:db.rollback(); log.exception('Segment save failed'); raise HTTPException(503,'could not save segment') from e
@app.post('/segments/{segment_id}/predict')
def predict(segment_id:int,horizon_days:Literal[7,15,30]=Query(30,alias='horizon'),db:Session=Depends(get_db)):
    row=db.get(RoadSegment,segment_id)
    if row is None:raise HTTPException(404,'segment not found')
    features={'pci':row.pci,'rainfall_mm':row.rainfall_mm,'traffic_count':row.traffic_count,'historical_damage_count':row.historical_damage_count,'pothole_count':row.pothole_count,'crack_area_ratio':row.crack_area_ratio,'days_since_inspection':max(0,(datetime.now(timezone.utc)-row.observed_at).days)}
    try:
        result=predict_one(features,horizon_days,settings.model_dir); db.add(Prediction(segment_id=row.id,horizon_days=horizon_days,predicted_pci=result['predicted_pci'],risk_score=result['risk_score'],recommendation=result['recommendation'],model_version='rf-horizon-v1')); db.commit(); return result
    except FileNotFoundError as e:raise HTTPException(503,str(e)) from e
    except Exception as e:db.rollback(); log.exception('Prediction failed'); raise HTTPException(500,'prediction failed') from e
@app.post('/detect')
async def detect_damage(image:UploadFile=File(...),segment_id:int|None=None,db:Session=Depends(get_db)):
    if not image.content_type or not image.content_type.startswith('image/'):raise HTTPException(415,'upload must be an image')
    from .detection import DamageDetector
    temp=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(image.filename or 'image.jpg').suffix or '.jpg') as f:
            temp=f.name; f.write(await image.read())
        detections=DamageDetector(settings.yolo_weights).predict(temp); saved=False
        if segment_id is not None:
            row=db.get(RoadSegment,segment_id)
            if row is None:raise HTTPException(404,'segment not found')
            import cv2
            frame=cv2.imread(temp)
            if frame is None:raise ValueError('Uploaded image could not be decoded')
            area=max(1,frame.shape[0]*frame.shape[1]); row.historical_damage_count+=len(detections)
            row.pothole_count+=sum('pothole' in d['label'].lower() for d in detections)
            row.crack_area_ratio=max(row.crack_area_ratio,min(1.0,sum(max(0,d['box_xyxy'][2]-d['box_xyxy'][0])*max(0,d['box_xyxy'][3]-d['box_xyxy'][1]) for d in detections)/area))
            if detections:row.damage_type=max((d['label'] for d in detections),key=lambda label:sum(x['label']==label for x in detections))
            try:
                from .severity import predict_severity
                features={'pci':row.pci,'rainfall_mm':row.rainfall_mm,'traffic_count':row.traffic_count,'historical_damage_count':row.historical_damage_count,'pothole_count':row.pothole_count,'crack_area_ratio':row.crack_area_ratio,'days_since_inspection':max(0,(datetime.now(timezone.utc)-row.observed_at).days)}
                row.severity=predict_severity(features,settings.model_dir/'severity.joblib')
            except FileNotFoundError:row.severity='high' if row.pci<45 or row.pothole_count>=4 else ('medium' if row.pci<70 or row.pothole_count>=2 else 'low')
            row.observed_at=datetime.now(timezone.utc)
            try:db.commit(); saved=True
            except Exception as e:db.rollback(); log.exception('Unable to persist image detections'); raise HTTPException(503,'could not persist detections') from e
        return {'detections':detections,'count':len(detections),'segment_id':segment_id,'saved_to_segment':saved,'source':'YOLO inference using configured weights'}
    except HTTPException:raise
    except (RuntimeError,FileNotFoundError,ValueError) as e:raise HTTPException(503,str(e)) from e
    except Exception as e:log.exception('Image detection failed'); raise HTTPException(500,'image inference failed') from e
    finally:
        if temp and os.path.exists(temp):os.unlink(temp)
@app.get('/segments/{segment_id}/predictions')
def history(segment_id:int,db:Session=Depends(get_db)):
    try:return [dict(horizon_days=p.horizon_days,predicted_pci=p.predicted_pci,risk_score=p.risk_score,recommendation=p.recommendation,created_at=p.created_at.isoformat() if p.created_at else None) for p in db.scalars(select(Prediction).where(Prediction.segment_id==segment_id).order_by(Prediction.created_at.desc())).all()]
    except Exception as e:log.exception('Prediction history query failed'); raise HTTPException(503,'database query failed') from e
