from pathlib import Path
import joblib

def risk_score(pci:float,rainfall_mm:float,traffic_count:int,historical_damage_count:int)->float:
    score=.55*(100-max(0,min(100,pci)))+.15*min(max(rainfall_mm,0),160)/160*100+.15*min(max(traffic_count,0),30000)/30000*100+.15*min(max(historical_damage_count,0),10)/10*100
    return round(float(min(100,max(0,score))),2)
def recommendation(pci:float,risk:float)->str:
    if pci<40 or risk>=75:return 'Urgent: schedule detailed inspection and rehabilitation'
    if pci<60 or risk>=55:return 'High: plan preventive treatment within 30 days'
    if pci<75 or risk>=35:return 'Moderate: monitor and schedule routine maintenance'
    return 'Low: continue routine inspection cycle'
def predict_one(record:dict,horizon:int,model_dir='models'):
    if horizon not in (7,15,30): raise ValueError('horizon must be 7, 15, or 30 days')
    path=Path(model_dir)/f'pci_{horizon}d.joblib'
    if not path.exists(): raise FileNotFoundError(f'Model missing: {path}; run python -m road_platform.cli train')
    import pandas as pd
    from .features import engineer_features,MODEL_FEATURES
    model=joblib.load(path); row=engineer_features(pd.DataFrame([record])); pci=float(max(0,min(100,model.predict(row[MODEL_FEATURES])[0])))
    risk=risk_score(pci,record.get('rainfall_mm',0),record.get('traffic_count',0),record.get('historical_damage_count',0))
    return {'predicted_pci':round(pci,2),'risk_score':risk,'recommendation':recommendation(pci,risk),'horizon_days':horizon}
