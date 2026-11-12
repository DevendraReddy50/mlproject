import logging
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from .config import settings
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='Atlas Roads',page_icon='🛣️',layout='wide')
st.title('🛣️ Intelligent Road Condition Assessment')
st.caption('Prototype dashboard · sample data is SYNTHETIC until connected to verified observations.')
@st.cache_data(ttl=45)
def get_segments():
    try:
        r=requests.get(settings.api_url+'/segments',timeout=5); r.raise_for_status(); data=r.json()
        if data:return pd.DataFrame(data)
    except requests.RequestException as e:logging.warning('API unavailable: %s',e)
    try:return pd.read_csv(settings.data_path)
    except (FileNotFoundError,pd.errors.EmptyDataError):return pd.DataFrame()
df=get_segments()
if df.empty:
    st.warning('No road data available. Run `python -m road_platform.cli data` to create explicitly synthetic prototype data.'); st.stop()
if 'severity' not in df:df['severity']='unknown'
if 'observed_at' in df:df['observed_at']=pd.to_datetime(df.observed_at,errors='coerce')
a,b,c=st.columns(3); a.metric('Road segments',f'{df.segment_id.nunique():,}' if 'segment_id' in df else f'{len(df):,}'); b.metric('Mean PCI',f"{df.pci.mean():.1f} / 100"); c.metric('High severity',int((df.severity=='high').sum()))
left,right=st.columns([1.3,1]);
with left:
    st.subheader('Condition distribution'); st.plotly_chart(px.histogram(df,x='pci',color='severity',nbins=20),use_container_width=True)
    st.subheader('Priority segments'); cols=[c for c in ('segment_id','road_name','pci','severity','rainfall_mm','traffic_count') if c in df]; st.dataframe(df[cols].sort_values('pci').head(100),use_container_width=True,hide_index=True)
with right:
    st.subheader('Road risk map'); center=[float(df.latitude.mean()),float(df.longitude.mean())] if {'latitude','longitude'}<=set(df) else [37.8,-122.4]; m=folium.Map(location=center,zoom_start=11)
    if {'latitude','longitude'}<=set(df):
        colors={'high':'red','medium':'orange','low':'green'}
        for _,r in df.dropna(subset=['latitude','longitude']).head(1000).iterrows():folium.CircleMarker([r.latitude,r.longitude],radius=5,color=colors.get(str(r.get('severity','')).lower(),'blue'),fill=True,fill_opacity=.75,popup=f"{r.get('road_name',r.get('segment_id','Road'))} · PCI {r.pci:.1f} · {r.get('severity','unknown')}").add_to(m)
    st_folium(m,use_container_width=True,height=450)
    st.subheader('Predictive maintenance'); st.write('Forecast service supports 7, 15, and 30 day horizons after models are trained.')
    if 'segment_id' in df:
        with st.form('forecast_form'):
            chosen=st.selectbox('Road segment',df.segment_id.drop_duplicates().tolist()); horizon=st.selectbox('Forecast horizon (days)',[7,15,30]); submitted=st.form_submit_button('Generate forecast')
        if submitted:
            try:
                response=requests.post(f'{settings.api_url}/segments/{int(chosen)}/predict',params={'horizon':horizon},timeout=20); response.raise_for_status(); st.json(response.json())
            except requests.RequestException as e:st.error(f'Forecast unavailable: {e}. Start the API and train models first.')
    st.info('Models trained on synthetic prototype data must not be used for operational decisions.')


