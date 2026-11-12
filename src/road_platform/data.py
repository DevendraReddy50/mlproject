"""Synthetic prototype data generator; replace with verified municipal records for production."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
log=logging.getLogger(__name__)
FEATURES=['pci','rainfall_mm','traffic_count','historical_damage_count','pothole_count','crack_area_ratio','days_since_inspection']
def generate_synthetic_data(n=1200, seed=42):
    rng=np.random.default_rng(seed); now=pd.Timestamp.now(tz='UTC').floor('D')
    pci=rng.uniform(20,100,n); rain=np.clip(rng.gamma(2,8,n),0,160); traffic=rng.integers(500,30000,n)
    hist=rng.poisson(2,n); potholes=rng.poisson(np.maximum(.2,(75-pci)/18),n)
    crack=np.clip((100-pci)/125+rng.normal(0,.08,n),0,1); age=rng.integers(1,400,n)
    damage=np.where(potholes>0,'pothole',np.where(crack>.4,'crack','none'))
    severity=np.where((pci<45)|(potholes>=4),'high',np.where((pci<70)|(potholes>=2),'medium','low'))
    # Simulated target: bounded future PCI degradation formula plus noise, explicitly not real-world evidence.
    rows=[]
    for h in (7,15,30):
        delta=(rain*.018+traffic/30000*.9+hist*.32+crack*2+potholes*.45+age/400*.55)*(h/30)+rng.normal(0,.65,n)
        rows.append(pd.DataFrame({'segment_id':np.arange(1,n+1),'road_name':[f'Prototype Road {i%60+1:02d}' for i in range(n)],'latitude':37.70+rng.random(n)*.24,'longitude':-122.52+rng.random(n)*.3,'pci':pci,'rainfall_mm':rain,'traffic_count':traffic,'historical_damage_count':hist,'pothole_count':potholes,'crack_area_ratio':crack,'days_since_inspection':age,'damage_type':damage,'severity':severity,'observed_at':now-pd.to_timedelta(age,unit='D'),'horizon_days':h,'future_pci':np.clip(pci-delta,0,100)}))
    return pd.concat(rows,ignore_index=True)
def load_data(path: str|Path, allow_synthetic=True):
    p=Path(path)
    if p.exists():
        df=pd.read_csv(p,parse_dates=['observed_at']); log.info('Loaded %d records from %s',len(df),p); return df
    if not allow_synthetic: raise FileNotFoundError(f'Dataset missing: {p}')
    log.warning('Using SYNTHETIC prototype data; this is not observed road data')
    p.parent.mkdir(parents=True,exist_ok=True); df=generate_synthetic_data(); df.to_csv(p,index=False); return df
