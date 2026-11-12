import logging
import pandas as pd
import plotly.express as px
log=logging.getLogger(__name__)
def summarize(df: pd.DataFrame)->dict:
    return {'rows':len(df),'columns':list(df.columns),'missing':df.isna().sum().to_dict(),'pci_summary':df.pci.describe().to_dict(),'severity_counts':df.severity.value_counts().to_dict() if 'severity' in df else {}}
def save_eda(df:pd.DataFrame, output='reports/eda.html'):
    from pathlib import Path
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    fig=px.histogram(df,x='pci',color='severity' if 'severity' in df else None,title='Road PCI distribution (prototype data)')
    fig.write_html(p); log.info('EDA chart saved to %s',p); return p
