"""Train horizon-specific regressors; scores are measured on a held-out split."""
import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from .features import engineer_features,MODEL_FEATURES
log=logging.getLogger(__name__)
def train_models(df,model_dir='models'):
    Path(model_dir).mkdir(parents=True,exist_ok=True); f=engineer_features(df); X=f[MODEL_FEATURES]
    reports={}
    for horizon in (7,15,30):
        part=f[f.horizon_days==horizon]
        if len(part)<10: raise ValueError(f'Need at least 10 rows for horizon {horizon}')
        a,b=train_test_split(np.arange(len(part)),test_size=.2,random_state=42)
        model=RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1)
        model.fit(engineer_features(part.iloc[a])[MODEL_FEATURES],part.iloc[a].future_pci)
        pred=model.predict(engineer_features(part.iloc[b])[MODEL_FEATURES]); y=part.iloc[b].future_pci
        reports[str(horizon)]={'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),'r2':float(r2_score(y,pred)),'test_rows':len(b)}
        joblib.dump(model,Path(model_dir)/f'pci_{horizon}d.joblib')
    # XGBoost is compared only when installed, trained/evaluated on the same real split.
    try:
        from xgboost import XGBRegressor
        part=f[f.horizon_days==30]; a,b=train_test_split(np.arange(len(part)),test_size=.2,random_state=42)
        xgb=XGBRegressor(n_estimators=120,max_depth=4,learning_rate=.05,objective='reg:squarederror',n_jobs=2,random_state=42)
        xgb.fit(engineer_features(part.iloc[a])[MODEL_FEATURES],part.iloc[a].future_pci); pr=xgb.predict(engineer_features(part.iloc[b])[MODEL_FEATURES]); y=part.iloc[b].future_pci
        reports['xgboost_30d']={'mae':float(mean_absolute_error(y,pr)),'rmse':float(np.sqrt(mean_squared_error(y,pr))),'r2':float(r2_score(y,pr)),'test_rows':len(b)}
        joblib.dump(xgb,Path(model_dir)/'xgboost_30d.joblib')
    except ImportError: log.warning('XGBoost unavailable; comparison skipped')
    log.info('Held-out evaluation: %s',reports); return reports
