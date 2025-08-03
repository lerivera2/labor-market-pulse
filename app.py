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
FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}

def get_series_ids(location="U.S. Total"):
    if location == "U.S. Total":
        return {"Job Openings": "JTS000000000000000JOL", "Unemployment Rate": "LNS14000000"}
    else:
        fips = STATE_FIPS[location]
        return {"Job Openings": f"JTS{fips}000000000JOL", "Unemployment Rate": f"LASST{fips}0000000000003"}

# --- Data Fetching and Processing ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_bls_data(series_ids, location):
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=24 * 30)).year
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": list(series_ids.values()), "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY})
    
    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers)
        response.raise_for_status()
        json_data = response.json()

        if json_data['status'] != 'REQUEST_SUCCEEDED':
            st.error(f"BLS API Error for {location}: {json_data.get('message', ['Unknown error'])[0]}")
            return None

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
    except Exception as e:
        st.error(f"An error occurred while fetching data for {location}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_all_states_latest_unemployment():
    series_ids = {f"LASST{fips}0000000000003": name for name, fips in STATE_FIPS.items()}
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=6*30)).year # Get last 6 months to ensure we have data
    
    headers = {'Content-type': 'application/json'}
    # BLS API has a limit of 50 series per request
    series_chunks = [list(series_ids.keys())[i:i + 50] for i in range(0, len(series_ids), 50)]
    
    latest_data = {}
    for chunk in series_chunks:
        data = json.dumps({"seriesid": chunk, "startyear": str(start_year), "endyear": str(end_year), "registrationkey": BLS_API_KEY, "latest": "true"})
        try:
            response = requests.post(BLS_API_URL, data=data, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            if json_data['status'] == 'REQUEST_SUCCEEDED':
                for series in json_data['Results']['series']:
                    if series['data']:
                        state_name = series_ids[series['seriesID']]
                        latest_data[state_name] = float(series['data'][0]['value'])
        except Exception:
            continue # Silently fail for a chunk if needed
            
    return pd.DataFrame(list(latest_data.items()), columns=['State', 'Unemployment Rate'])

# --- Visualization Functions ---
def create_choropleth_map(df):
    fig = px.choropleth(
        df,
        locations=df['State'],
        locationmode="USA-states",
        color='Unemployment Rate',
        color_continuous_scale="Plasma",
        scope="usa",
        title="Latest Unemployment Rate by State",
        labels={'Unemployment Rate': 'Rate (%)'}
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title_x=0.5,
        clickmode='event+select'
    )
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

# Initialize session state for location
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = "U.S. Total"

st.title("Labor-Market Pulse")
st.markdown("An interactive dashboard tracking U.S. labor market health via Job Openings and Unemployment data.")

# --- Sidebar ---
with st.sidebar:
    st.header("Controls & Information")
    location_list = ["U.S. Total"] + sorted(STATE_FIPS.keys())
    
    # Update session state if selectbox changes
    def update_location_from_select():
        st.session_state.selected_location = st.session_state.selectbox_location
    
    st.selectbox(
        "Select a Location:",
        location_list,
        key='selectbox_location',
        on_change=update_location_from_select,
        index=location_list.index(st.session_state.selected_location)
    )

    if st.button('Refresh All Data'):
        st.cache_data.clear()
        st.experimental_rerun()

    with st.expander("Design & Accessibility Notes"):
        st.markdown("""
        - **Visual Integrity:** Charts prioritize data clarity, maximizing the data-ink ratio by minimizing gridlines and decorative elements (Tufte).
        - **Color Choice:** The map uses a sequential `Plasma` palette, which is perceptually uniform and colorblind-safe. The line chart uses a distinct, high-contrast palette.
        - **Layout:** The dashboard follows a Z-pattern, placing high-level KPIs and the map at the top, with the detailed time-series chart below for deeper analysis.
        - **Accessibility:** High-contrast labels and clear typography are used throughout. Interactive elements like tooltips provide data access without relying on color alone.
        """)
    st.sidebar.info("Data Source: U.S. Bureau of Labor Statistics (BLS). Dashboard by Gemini.")

# --- Main Dashboard Area ---
# Top Row: KPIs and Map
kpi_col, map_col = st.columns([1, 2])

# Fetch data for the selected location (KPIs and Line Chart)
series_ids = get_series_ids(st.session_state.selected_location)
location_data_df = get_bls_data(series_ids, st.session_state.selected_location)

with kpi_col:
    st.subheader(f"Key Metrics for {st.session_state.selected_location}")
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
        # Use st.plotly_chart with on_click event
        selected_point = st.plotly_chart(map_fig, use_container_width=True, on_select="rerun")
        
        # Update session state if a state on the map is clicked
        if selected_point and selected_point.selection and selected_point.selection['points']:
            clicked_location = selected_point.selection['points'][0]['location']
            if clicked_location in STATE_FIPS.keys():
                 if st.session_state.selected_location != clicked_location:
                    st.session_state.selected_location = clicked_location
                    st.experimental_rerun()

    else:
        st.warning("Could not load map data.")

# Bottom Row: Time-Series Chart
st.markdown("---")
if location_data_df is not None and not location_data_df.empty:
    time_series_fig = create_time_series_chart(location_data_df, st.session_state.selected_location)
    st.plotly_chart(time_series_fig, use_container_width=True)
else:
    st.error(f"Could not display historical data for {st.session_state.selected_location}. Please select another location.")

