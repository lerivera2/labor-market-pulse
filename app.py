# Labor-Market Pulse: An Advanced Dashboard
# This script fetches, processes, and visualizes U.S. and State-level labor market data,
# adhering to professional data visualization best practices.

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Labor-Market Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BLS API Configuration ---
BLS_API_KEY = os.environ.get("BLS_API_KEY")
if not BLS_API_KEY:
    try:
        BLS_API_KEY = st.secrets["BLS_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("A BLS_API_KEY is required. Please set it as a Streamlit secret.")
        st.stop()
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# --- Data Mappings ---
STATE_MAPPING = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 
    'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
STATE_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06', 'Colorado': '08', 
    'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 
    'Hawaii': '15', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 
    'Kentucky': '21', 'Louisiana': '22', 'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 
    'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 
    'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35', 
    'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40', 
    'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46', 
    'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '52', 
    'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
}
INDUSTRY_CODES = {
    "Total Nonfarm": "000000", "Construction": "200000", "Manufacturing": "300000",
    "Trade, Transportation, and Utilities": "400000", "Information": "510000",
    "Financial Activities": "520000", "Professional and Business Services": "600000",
    "Education and Health Services": "620000", "Leisure and Hospitality": "700000"
}
MSA_CODES = {
    "New York-Newark-Jersey City, NY-NJ-PA": "35620", "Los Angeles-Long Beach-Anaheim, CA": "31080",
    "Chicago-Naperville-Elgin, IL-IN-WI": "16980", "Dallas-Fort Worth-Arlington, TX": "19100",
    "Houston-The Woodlands-Sugar Land, TX": "26420", "Washington-Arlington-Alexandria, DC-VA-MD-WV": "47900",
    "Miami-Fort Lauderdale-Pompano Beach, FL": "33100", "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD": "37980",
    "Atlanta-Sandy Springs-Alpharetta, GA": "12060", "Boston-Cambridge-Newton, MA-NH": "14460"
}

def get_series_ids(loc_type, location, industry):
    series = {}
    if loc_type == "U.S. Total":
        ind_code = INDUSTRY_CODES[industry]
        series["Job Openings"] = f"JTS{ind_code}000000000JOL"
        if industry == "Total Nonfarm":
            series["Unemployment Rate"] = "LNS14000000"
            series["Quits Rate"] = "JTS000000000000000QUL"
    elif loc_type == "State":
        fips = STATE_FIPS.get(location)
        if fips:
            series["Unemployment Rate"] = f"LASST{fips}0000000000003"
            if industry == "Total Nonfarm":
                series["Job Openings"] = f"JTS{fips}000000000JOL"
                series["Quits Rate"] = f"JTS{fips}000000000QUL"
    elif loc_type == "Metropolitan Area":
        msa_code = MSA_CODES.get(location)
        if msa_code:
            series["Unemployment Rate"] = f"LAUMT{msa_code}00000000003"
    return series

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def get_bls_data(series_ids, location):
    if not series_ids: return None
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=25 * 30)).year
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": list(series_ids.values()), "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY})
    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers, timeout=20)
        response.raise_for_status()
        json_data = response.json()
        if json_data['status'] != 'REQUEST_SUCCEEDED': return None
        all_series_data = []
        for series_name, series_id in series_ids.items():
            for result_series in json_data['Results']['series']:
                if result_series['seriesID'] == series_id and result_series['data']:
                    df = pd.DataFrame(result_series['data'])
                    df['date'] = pd.to_datetime(df['year'] + '-' + df['periodName'])
                    df.set_index('date', inplace=True)
                    df['value'] = pd.to_numeric(df['value'])
                    df.rename(columns={'value': series_name}, inplace=True)
                    all_series_data.append(df[[series_name]])
                    break
        if not all_series_data: return None
        return pd.concat(all_series_data, axis=1).sort_index().last('25M').ffill()
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=3600)
def get_all_states_latest_unemployment():
    series_ids = {f"LASST{fips}0000000000003": name for name, fips in STATE_FIPS.items()}
    end_year, start_year = date.today().year, (date.today() - timedelta(days=12*30)).year
    headers = {'Content-type': 'application/json'}
    series_chunks = [list(series_ids.keys())[i:i + 50] for i in range(0, len(series_ids), 50)]
    latest_data = {}
    for chunk in series_chunks:
        data = json.dumps({"seriesid": chunk, "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY})
        try:
            response = requests.post(BLS_API_URL, data=data, headers=headers, timeout=20)
            response.raise_for_status()
            json_data = response.json()
            if json_data['status'] == 'REQUEST_SUCCEEDED':
                for series in json_data['Results']['series']:
                    if series['data']:
                        state_name = series_ids[series['seriesID']]
                        latest_value = pd.to_numeric(pd.DataFrame(series['data'])['value']).iloc[-1]
                        latest_data[state_name] = latest_value
        except requests.exceptions.RequestException: continue
    if not latest_data: return None
    df = pd.DataFrame(list(latest_data.items()), columns=['State', 'Unemployment Rate'])
    df['State_Abbr'] = df['State'].map(STATE_MAPPING)
    return df

# --- Visualization Functions ---
def create_choropleth_map(df):
    fig = px.choropleth(df, locations='State_Abbr', locationmode="USA-states", color='Unemployment Rate',
                        color_continuous_scale="Plasma", scope="usa", hover_name='State',
                        title="Latest Unemployment Rate by State", labels={'Unemployment Rate': 'Rate (%)'})
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5)
    return fig

def create_time_series_chart(df, location, metrics):
    chart_df = df.last('24M')
    fig = go.Figure()
    colors = {'openings': '#1f77b4', 'unemployment': '#ff7f0e', 'quits': '#2ca02c', 'grid': '#e0e0e0'}
    
    if 'Job Openings' in metrics:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Job Openings'] / 1000, name='Job Openings (K)', mode='lines+markers', line=dict(color=colors['openings'], width=2.5), hovertemplate='<b>Job Openings:</b> %{y:,.0f}K<extra></extra>'))
    if 'Unemployment Rate' in metrics:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Unemployment Rate'], name='Unemployment Rate (%)', mode='lines+markers', line=dict(color=colors['unemployment'], width=2.5, dash='dash'), yaxis='y2', hovertemplate='<b>Unemployment Rate:</b> %{y:.1f}%<extra></extra>'))

    fig.update_layout(title=dict(text=f'<b>Openings vs. Unemployment for {location}</b>', font_size=18, x=0.5),
                      xaxis=dict(title_text=None, showgrid=False),
                      yaxis=dict(title=dict(text='Job Openings (in thousands)', font=dict(color=colors['openings'])), tickfont=dict(color=colors['openings']), showgrid=True, gridcolor=colors['grid']),
                      yaxis2=dict(title=dict(text='Unemployment Rate (%)', font=dict(color=colors['unemployment'])), tickfont=dict(color=colors['unemployment']), overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      template='simple_white', height=350, margin=dict(l=50, r=50, t=50, b=20), hovermode='x unified')
    return fig

def create_quits_rate_chart(df, location):
    chart_df = df.last('24M')
    fig = go.Figure()
    colors = {'quits': '#2ca02c', 'grid': '#e0e0e0'}
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Quits Rate'], name='Quits Rate (%)', mode='lines+markers', line=dict(color=colors['quits'], width=2.5), hovertemplate='<b>Quits Rate:</b> %{y:.1f}%<extra></extra>'))
    fig.update_layout(title=dict(text=f'<b>Quits Rate Trend for {location}</b>', font_size=18, x=0.5),
                      xaxis=dict(title_text='Date', showgrid=False),
                      yaxis=dict(title=dict(text='Quits Rate (%)', font=dict(color=colors['quits'])), tickfont=dict(color=colors['quits']), showgrid=True, gridcolor=colors['grid']),
                      template='simple_white', height=300, margin=dict(l=50, r=50, t=50, b=50), hovermode='x unified')
    return fig

# --- Streamlit App Layout ---
st.title("Labor-Market Pulse")
st.markdown("An interactive dashboard tracking U.S. labor market health via Job Openings, Unemployment, and Quits data.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls & Information")
    loc_type = st.radio("Select Location Type:", ["U.S. Total", "State", "Metropolitan Area"], horizontal=True)
    
    if loc_type == "U.S. Total":
        selected_location = "U.S. Total"
        industry_list = list(INDUSTRY_CODES.keys())
        selected_industry = st.selectbox("Select an Industry (for Job Openings):", industry_list)
    elif loc_type == "State":
        selected_location = st.selectbox("Select a State:", sorted(STATE_FIPS.keys()))
        selected_industry = "Total Nonfarm" # Industry data not available at state level via this API structure
    else: # Metropolitan Area
        selected_location = st.selectbox("Select a Metro Area:", sorted(MSA_CODES.keys()))
        selected_industry = "Total Nonfarm"

    if st.button('Refresh All Data'):
        st.cache_data.clear()
        st.rerun()
    with st.expander("Design & Accessibility Notes"):
        st.markdown("""
        - **Visual Integrity:** Charts prioritize data clarity by minimizing non-data elements.
        - **Color Choice:** The map uses a colorblind-safe sequential palette. Charts use high-contrast colors.
        - **Layout:** A Z-pattern places high-level KPIs and the map at the top, with details below.
        """)
    st.sidebar.info("Data Source: U.S. Bureau of Labor Statistics (BLS). Dashboard by Gemini.")

# --- Main Dashboard Area ---
series_ids = get_series_ids(loc_type, selected_location, selected_industry)
location_data_df = get_bls_data(series_ids, selected_location)

kpi_col, map_col = st.columns([1, 2])

with kpi_col:
    st.subheader(f"Key Metrics for {selected_location}")
    if selected_industry != "Total Nonfarm":
        st.caption(f"Industry: {selected_industry}")

    if location_data_df is not None and len(location_data_df) > 1:
        latest_date = location_data_df.index[-1].strftime('%B %Y')
        previous_date = location_data_df.index[-2].strftime('%B %Y')

        for metric in ["Unemployment Rate", "Job Openings", "Quits Rate"]:
            if metric in location_data_df.columns:
                latest_val = location_data_df[metric].iloc[-1]
                prev_val = location_data_df[metric].iloc[-2]
                delta = latest_val - prev_val
                
                if metric == "Unemployment Rate":
                    st.metric(label=f"{metric} ({latest_date})", value=f"{latest_val}%", delta=f"{delta:.2f}%", delta_color="inverse", help=f"Change from {previous_date}. Lower is better.")
                elif metric == "Job Openings":
                    st.metric(label=f"{metric} ({latest_date})", value=f"{latest_val/1_000_000:.2f}M" if latest_val >= 1_000_000 else f"{latest_val/1000:,.0f}K", delta=f"{delta/1000:,.1f}K", delta_color="normal", help=f"Change from {previous_date}. Higher indicates a tighter labor market.")
                elif metric == "Quits Rate":
                    st.metric(label=f"{metric} ({latest_date})", value=f"{latest_val}%", delta=f"{delta:.2f}%", delta_color="normal", help=f"Change from {previous_date}. Higher indicates strong worker confidence.")
    else:
        st.warning("Data not available for the selected filters.")

with map_col:
    all_states_df = get_all_states_latest_unemployment()
    if all_states_df is not None and not all_states_df.empty:
        st.plotly_chart(create_choropleth_map(all_states_df), use_container_width=True)
    else:
        st.warning("Could not load map data.")

st.markdown("---")

if location_data_df is not None:
    # Chart 1: Openings vs Unemployment
    metrics_for_chart1 = [m for m in ["Job Openings", "Unemployment Rate"] if m in location_data_df.columns]
    if len(metrics_for_chart1) == 2:
        st.plotly_chart(create_time_series_chart(location_data_df, selected_location, metrics_for_chart1), use_container_width=True)
    
    # Chart 2: Quits Rate
    if 'Quits Rate' in location_data_df.columns:
        st.plotly_chart(create_quits_rate_chart(location_data_df, selected_location), use_container_width=True)
    
    if not metrics_for_chart1 and 'Quits Rate' not in location_data_df.columns:
         st.error(f"Could not display historical data for {selected_location}. The selected data series may be unavailable.")
else:
    st.error(f"Could not retrieve any historical data for {selected_location}.")
