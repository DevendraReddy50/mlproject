import logging
import pandas as pd
from .data import FEATURES
log=logging.getLogger(__name__)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required={'segment_id','pci','rainfall_mm','traffic_count','historical_damage_count','observed_at'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    out=df.copy().drop_duplicates(subset=['segment_id','observed_at','horizon_days'] if 'horizon_days' in df else ['segment_id','observed_at'])
    for c in ('pci','rainfall_mm','traffic_count','historical_damage_count'):
        out[c]=pd.to_numeric(out[c],errors='coerce')
    out=out.dropna(subset=['pci','rainfall_mm','traffic_count','historical_damage_count'])
    out=out[out.pci.between(0,100)&out.rainfall_mm.ge(0)&out.traffic_count.ge(0)]
    for c in FEATURES:
        if c not in out: out[c]=0
        out[c]=pd.to_numeric(out[c],errors='coerce').fillna(out[c].median() if out[c].notna().any() else 0)
    out['observed_at']=pd.to_datetime(out.observed_at,utc=True,errors='coerce'); out=out.dropna(subset=['observed_at'])
    log.info('Cleaned dataset: %d rows',len(out))
    if out.empty: raise ValueError('No usable rows after cleaning')
    return out.reset_index(drop=True)
