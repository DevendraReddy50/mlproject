import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from .data import FEATURES
log=logging.getLogger(__name__)
def train_severity(df:pd.DataFrame,model_dir='models'):
    if df.severity.nunique()<2: raise ValueError('Severity training requires at least two classes')
    # Horizon copies represent the same inspection; keep each segment in one split.
    df=df.drop_duplicates('segment_id').copy() if 'segment_id' in df else df.copy()
    X=df[FEATURES]; y=df.severity
    strat=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    m=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xtr,ytr)
    metrics={'accuracy':float(accuracy_score(yte,m.predict(Xte))),'report':classification_report(yte,m.predict(Xte),output_dict=True,zero_division=0),'test_rows':len(yte)}
    Path(model_dir).mkdir(parents=True,exist_ok=True); joblib.dump(m,Path(model_dir)/'severity.joblib'); log.info('Severity held-out metrics: %s',metrics['accuracy']); return metrics
def predict_severity(features:dict,model_path='models/severity.joblib'):
    m=joblib.load(model_path); return str(m.predict(pd.DataFrame([features])[FEATURES])[0])


