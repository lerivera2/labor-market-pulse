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

# --- State and Series ID Mappings ---
STATE_MAPPING = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
STATE_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06',
    'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11', 'Florida': '12',
    'Georgia': '13', 'Hawaii': '15', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
    'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23', 'Maryland': '24',
    'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29',
    'Montana': '30', 'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34',
    'New Mexico': '35', 'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
    'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45',
    'South Dakota': '46', 'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50',
    'Virginia': '51', 'Washington': '52', 'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
}

def get_series_ids(location="U.S. Total"):
    if location == "U.S. Total":
        return {"Job Openings": "JTS000000000000000JOL", "Unemployment Rate": "LNS14000000"}
    else:
        fips = STATE_FIPS.get(location)
        if fips:
            return {"Job Openings": f"JTS{fips}000000000JOL", "Unemployment Rate": f"LASST{fips}0000000000003"}
        return None

# --- Data Fetching and Processing ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_bls_data(series_ids, location):
    if not series_ids: return None
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=24 * 30)).year
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": list(series_ids.values()), "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY})

    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers, timeout=15)
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
        combined_df = pd.concat(all_series_data, axis=1).sort_index()
        return combined_df.last('24M').ffill()
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=3600)
def get_all_states_latest_unemployment():
    series_ids = {f"LASST{fips}0000000000003": name for name, fips in STATE_FIPS.items()}
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=12*30)).year # Get last year of data to be safe

    headers = {'Content-type': 'application/json'}
    series_chunks = [list(series_ids.keys())[i:i + 50] for i in range(0, len(series_ids), 50)]
    
    latest_data = {}
    for chunk in series_chunks:
        data = json.dumps({"seriesid": chunk, "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY})
        try:
            response = requests.post(BLS_API_URL, data=data, headers=headers, timeout=15)
            response.raise_for_status()
            json_data = response.json()
            if json_data['status'] == 'REQUEST_SUCCEEDED':
                for series in json_data['Results']['series']:
                    if series['data']:
                        state_name = series_ids[series['seriesID']]
                        # Create a mini-dataframe to find the actual latest value
                        temp_df = pd.DataFrame(series['data'])
                        temp_df['value'] = pd.to_numeric(temp_df['value'])
                        latest_value = temp_df.iloc[-1]['value'] # Last item is the latest
                        latest_data[state_name] = latest_value
        except requests.exceptions.RequestException:
            continue
            
    if not latest_data: return None
    
    df = pd.DataFrame(list(latest_data.items()), columns=['State', 'Unemployment Rate'])
    df['State_Abbr'] = df['State'].map(STATE_MAPPING)
    return df

# --- Visualization Functions ---
def create_choropleth_map(df):
    fig = px.choropleth(
        df,
        locations='State_Abbr',
        locationmode="USA-states",
        color='Unemployment Rate',
        color_continuous_scale="Plasma",
        scope="usa",
        hover_name='State',
        title="Latest Unemployment Rate by State",
        labels={'Unemployment Rate': 'Rate (%)'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title_x=0.5)
    return fig

def create_time_series_chart(df, location):
    fig = go.Figure()
    colors = {'openings': '#1f77b4', 'unemployment': '#ff7f0e', 'grid': '#e0e0e0'}

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Job Openings'] / 1000, name='Job Openings (in thousands)',
        mode='lines+markers', line=dict(color=colors['openings'], width=2.5),
        hovertemplate='<b>Job Openings:</b> %{y:,.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Unemployment Rate'], name='Unemployment Rate (%)',
        mode='lines+markers', line=dict(color=colors['unemployment'], width=2.5, dash='dash'),
        yaxis='y2', hovertemplate='<b>Unemployment Rate:</b> %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f'<b>Historical Trend for {location}</b>', font_size=18, x=0.5),
        xaxis=dict(title_text='Date', showgrid=False),
        yaxis=dict(title=dict(text='Job Openings (in thousands)', font=dict(color=colors['openings'])), tickfont=dict(color=colors['openings']), showgrid=True, gridcolor=colors['grid']),
        yaxis2=dict(title=dict(text='Unemployment Rate (%)', font=dict(color=colors['unemployment'])), tickfont=dict(color=colors['unemployment']), overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='simple_white', height=400, margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    return fig

# --- Streamlit App Layout ---
st.title("Labor-Market Pulse")
st.markdown("An interactive dashboard tracking U.S. labor market health via Job Openings and Unemployment data.")

with st.sidebar:
    st.header("Controls & Information")
    location_list = ["U.S. Total"] + sorted(STATE_FIPS.keys())
    selected_location = st.selectbox("Select a Location:", location_list)
    if st.button('Refresh All Data'):
        st.cache_data.clear()
        st.rerun()
    with st.expander("Design & Accessibility Notes"):
        st.markdown("""
        - **Visual Integrity:** Charts prioritize data clarity by minimizing non-data elements.
        - **Color Choice:** The map uses a colorblind-safe sequential palette. The line chart uses high-contrast colors.
        - **Layout:** A Z-pattern places high-level KPIs and the map at the top, with details below.
        - **Accessibility:** High-contrast labels and tooltips provide data access without relying on color alone.
        """)
    st.sidebar.info("Data Source: U.S. Bureau of Labor Statistics (BLS). Dashboard by Gemini.")

kpi_col, map_col = st.columns([1, 2])

series_ids = get_series_ids(selected_location)
location_data_df = get_bls_data(series_ids, selected_location)

with kpi_col:
    st.subheader(f"Key Metrics for {selected_location}")
    if location_data_df is not None and not location_data_df.empty:
        latest_date = location_data_df.index[-1].strftime('%B %Y')
        latest_openings = location_data_df['Job Openings'].iloc[-1]
        latest_unemployment = location_data_df['Unemployment Rate'].iloc[-1]
        st.metric(label=f"Unemployment Rate ({latest_date})", value=f"{latest_unemployment}%")
        st.metric(label=f"Job Openings ({latest_date})", value=f"{latest_openings/1_000_000:.2f}M" if latest_openings >= 1_000_000 else f"{latest_openings/1000:,.0f}K")
    else:
        st.warning("Data not available for selected KPIs.")

with map_col:
    all_states_df = get_all_states_latest_unemployment()
    if all_states_df is not None and not all_states_df.empty:
        map_fig = create_choropleth_map(all_states_df)
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.warning("Could not load map data.")

st.markdown("---")
if location_data_df is not None and not location_data_df.empty:
    time_series_fig = create_time_series_chart(location_data_df, selected_location)
    st.plotly_chart(time_series_fig, use_container_width=True)
else:
    st.error(f"Could not display historical data for {selected_location}. Please select another location.")
