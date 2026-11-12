import pandas as pd
import pytest
from road_platform.data import generate_synthetic_data
from road_platform.preprocessing import clean_data
from road_platform.prediction import risk_score,recommendation
from road_platform.features import engineer_features,MODEL_FEATURES

def test_generator_is_explicitly_structured():
    df=generate_synthetic_data(30,1); assert len(df)==90; assert set(df.horizon_days)=={7,15,30}; assert df.future_pci.between(0,100).all()
def test_clean_data_removes_invalid_rows():
    df=generate_synthetic_data(10); df.loc[0,'pci']=101; out=clean_data(df); assert out.pci.between(0,100).all()
def test_feature_engineering_produces_training_columns():
    df=clean_data(generate_synthetic_data(10)); out=engineer_features(df); assert set(MODEL_FEATURES)<=set(out)
def test_risk_and_recommendation_bounds():
    assert risk_score(100,0,0,0)==0; assert risk_score(0,160,30000,10)==100; assert recommendation(20,90).startswith('Urgent')
def test_clean_data_rejects_missing_required_columns():
    with pytest.raises(ValueError):clean_data(pd.DataFrame({'pci':[55]}))
