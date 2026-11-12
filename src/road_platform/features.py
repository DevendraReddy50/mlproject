import pandas as pd
from .data import FEATURES
def engineer_features(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if 'observed_at' in out:
        dt=pd.to_datetime(out.observed_at,utc=True); out['observation_month']=dt.dt.month; out['observation_dayofyear']=dt.dt.dayofyear
    out['traffic_per_pci']=out.traffic_count/(out.pci.clip(lower=1))
    out['rain_damage_interaction']=out.rainfall_mm*out.historical_damage_count
    out['damage_pressure']=.4*(100-out.pci)+.2*out.rainfall_mm.clip(upper=100)+.2*out.historical_damage_count*10+.2*out.traffic_count/300
    return out
MODEL_FEATURES=FEATURES+['traffic_per_pci','rain_damage_interaction','damage_pressure']
