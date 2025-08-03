# Labor-Market Pulse: Professional Edition
# Enhanced version with improved readability and user experience, incorporating user-provided design.

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
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #E0E0E0;
    margin-bottom: 1rem;
    text-align: center;
    height: 100%;
}

[data-theme="dark"] .metric-card {
    background-color: #262730;
    border: 1px solid #444;
}

.metric-label {
    font-size: 0.9rem;
    color: #6c757d;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-delta {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# --- BLS API Configuration ---
BLS_API_KEY = os.environ.get("BLS_API_KEY") or st.secrets.get("BLS_API_KEY")
if not BLS_API_KEY:
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

# --- Helper Functions ---
def get_series_ids(loc_type, location, industry):
    series = {}
    if loc_type == "U.S. Total":
        ind_code = INDUSTRY_CODES[industry]
        series["Job Openings"] = f"JTS{ind_code}000000000JOL"
        if industry == "Total Nonfarm":
            series["Unemployment Rate"] = "LNS14000000"
            series["Quits Rate"] = "JTS000000000000000QUR" # Corrected ID for Rate
    elif loc_type == "State":
        fips = STATE_FIPS.get(location)
        if fips:
            series["Unemployment Rate"] = f"LASST{fips}0000000000003"
            if industry == "Total Nonfarm":
                series["Job Openings"] = f"JTS{fips}000000000JOL"
                series["Quits Rate"] = f"JTS{fips}000000000QUR" # Corrected ID for Rate
    elif loc_type == "Metropolitan Area":
        msa_code = MSA_CODES.get(location)
        if msa_code:
            series["Unemployment Rate"] = f"LAUMT{msa_code}00000000003"
    return series

@st.cache_data(ttl=3600)
def get_bls_data(series_ids, years_to_fetch=5):
    if not series_ids: return None
    end_year, start_year = date.today().year, date.today().year - years_to_fetch
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
        return pd.concat(all_series_data, axis=1).sort_index().ffill()
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

# --- Visualization Components ---
def create_choropleth_map(df):
    fig = px.choropleth(df, locations='State_Abbr', locationmode="USA-states", color='Unemployment Rate',
                        color_continuous_scale="Viridis", scope="usa", hover_name='State',
                        title="Latest Unemployment Rate by State", labels={'Unemployment Rate': 'Rate (%)'})
    fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), title_x=0.5, geo=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

def create_time_series_chart(df, location, metrics):
    fig = go.Figure()
    colors = {'openings': '#004A7F', 'unemployment': '#FF7F0E', 'grid': '#ddd'}
    if 'Job Openings' in metrics:
        fig.add_trace(go.Scatter(x=df.index, y=df['Job Openings'] / 1000, name='Job Openings (K)', mode='lines', line=dict(color=colors['openings'], width=2.5)))
    if 'Unemployment Rate' in metrics:
        fig.add_trace(go.Scatter(x=df.index, y=df['Unemployment Rate'], name='Unemployment Rate (%)', mode='lines', line=dict(color=colors['unemployment'], width=2.5, dash='dash'), yaxis='y2'))
    fig.update_layout(title=dict(text=f'<b>Openings vs. Unemployment for {location}</b>', font_size=16, x=0.5),
                      xaxis=dict(title_text=None, showgrid=False),
                      yaxis=dict(title=dict(text='Job Openings (K)'), showgrid=True, gridcolor=colors['grid']),
                      yaxis2=dict(title=dict(text='Unemployment (%)'), overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      template='plotly_white', height=350, margin=dict(l=50, r=50, t=50, b=20), hovermode='x unified')
    return fig

def create_quits_rate_chart(df, location):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Quits Rate'], name='Quits Rate (%)', mode='lines', line=dict(color='#2ca02c', width=2.5)))
    fig.update_layout(title=dict(text=f'<b>Quits Rate Trend for {location}</b>', font_size=16, x=0.5),
                      xaxis=dict(title_text='Date', showgrid=False),
                      yaxis=dict(title=dict(text='Quits Rate (%)'), showgrid=True, gridcolor='#ddd'),
                      template='plotly_white', height=300, margin=dict(l=50, r=50, t=50, b=50), hovermode='x unified')
    return fig

def create_sparkline(df, metric):
    spark = go.Figure(go.Scatter(x=df.index[-12:], y=df[metric].iloc[-12:], mode='lines', line=dict(width=2)))
    spark.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=50, xaxis_visible=False, yaxis_visible=False, plot_bgcolor='rgba(0,0,0,0)')
    return spark

def create_gauge_chart(current, max_hist):
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=current,
        gauge={'axis':{'range':[0, max_hist]}, 'bar':{'color':'#004A7F'}},
        title={'text':'Current Rate vs. Historical High'}
    ))
    gauge.update_layout(height=250, margin=dict(l=30,r=30,t=60,b=30))
    return gauge

# --- State Management ---
def initialize_session_state():
    if 'init' not in st.session_state:
        st.session_state.loc_type = "U.S. Total"
        st.session_state.selected_location = "U.S. Total"
        st.session_state.selected_industry = "Total Nonfarm"
        st.session_state.base_month = None
        st.session_state.init = True

def reset_to_defaults():
    st.session_state.loc_type = "U.S. Total"
    st.session_state.selected_location = "U.S. Total"
    st.session_state.selected_industry = "Total Nonfarm"
    st.session_state.base_month = None

initialize_session_state()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls & Information")
    base_data_df = get_bls_data(get_series_ids("U.S. Total", "U.S. Total", "Total Nonfarm"))
    if base_data_df is not None:
        min_date, max_date = base_data_df.index.min(), base_data_df.index.max()
        if st.session_state.base_month is None:
            st.session_state.base_month = max_date.to_pydatetime()
        
        selected_date = st.slider("Select Base Month:", 
                                  min_value=min_date.to_pydatetime(), 
                                  max_value=max_date.to_pydatetime(), 
                                  value=st.session_state.base_month, 
                                  format="MMM YYYY")
        st.session_state.base_month = selected_date

    st.radio("Location Type:", ["U.S. Total", "State", "Metropolitan Area"], key='loc_type', horizontal=True)
    if st.session_state.loc_type == "State":
        st.selectbox("State:", sorted(STATE_FIPS.keys()), key='selected_location')
    elif st.session_state.loc_type == "Metropolitan Area":
        st.selectbox("Metro Area:", sorted(MSA_CODES.keys()), key='selected_location')
    else: # US Total
        st.session_state.selected_location = "U.S. Total"
        st.selectbox("Industry:", list(INDUSTRY_CODES.keys()), key='selected_industry')

    col1, col2 = st.columns(2)
    if col1.button('Refresh All Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    col2.button('Reset to Defaults', on_click=reset_to_defaults, use_container_width=True)
    with st.expander("Design & Accessibility Notes"):
        st.markdown("- **Visual Integrity:** Minimize non-data ink.\n- **Color Choice:** Sequential, colorblind-safe palettes.\n- **Layout:** Z-pattern: top KPIs/map, then details.")
    st.info("Data Source: U.S. Bureau of Labor Statistics (BLS)")

# --- Main Dashboard Area ---
st.markdown('<h1 class="main-title">Labor Market Pulse</h1>', unsafe_allow_html=True)
loc_title = st.session_state.selected_location
if st.session_state.loc_type == "U.S. Total" and st.session_state.selected_industry != "Total Nonfarm":
    loc_title += f" ({st.session_state.selected_industry})"
st.header(f"Dashboard for {loc_title}")

series_ids = get_series_ids(st.session_state.loc_type, st.session_state.selected_location, st.session_state.selected_industry if st.session_state.loc_type == "U.S. Total" else "Total Nonfarm")
full_data_df = get_bls_data(series_ids)

if full_data_df is None:
    st.error("Could not retrieve data for the selected filters. Please try a different selection.")
    st.stop()

display_data_df = full_data_df[full_data_df.index <= pd.to_datetime(st.session_state.base_month)]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ State Map", "📈 Historical Trends", "📋 Data Export"])

with tab1:
    st.subheader("Key Performance Indicators")
    if display_data_df is not None and len(display_data_df) > 1:
        latest_date_str = display_data_df.index[-1].strftime('%b %Y')
        previous_date_str = display_data_df.index[-2].strftime('%B %Y')
        
        metrics_to_show = ["Unemployment Rate", "Job Openings", "Quits Rate"]
        available_metrics = [m for m in metrics_to_show if m in display_data_df.columns]
        cols = st.columns(len(available_metrics) or 1)

        for i, metric in enumerate(available_metrics):
            with cols[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                latest_val = display_data_df[metric].iloc[-1]
                delta = latest_val - display_data_df[metric].iloc[-2]
                
                # Custom HTML for metric display
                st.markdown(f'<div class="metric-label">{metric} ({latest_date_str})</div>', unsafe_allow_html=True)
                if metric == "Job Openings":
                    value_str = f"{latest_val/1e6:.2f}M" if latest_val >= 1e6 else f"{latest_val/1e3:,.0f}K"
                    delta_str = f"{delta/1e3:,.1f}K"
                else:
                    value_str = f"{latest_val:.1f}%"
                    delta_str = f"{delta:+.2f}%"
                
                delta_color = "green" if (delta > 0 and metric != "Unemployment Rate") or (delta < 0 and metric == "Unemployment Rate") else "red"
                st.markdown(f'<div class="metric-value">{value_str}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-delta" style="color:{delta_color};">{delta_str} vs {previous_date_str}</div>', unsafe_allow_html=True)
                
                st.plotly_chart(create_sparkline(display_data_df, metric), use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Not enough data to display KPIs.")

    st.markdown("---")
    if "Unemployment Rate" in display_data_df.columns:
        current = display_data_df['Unemployment Rate'].iloc[-1]
        max_hist = display_data_df['Unemployment Rate'].max()
        st.plotly_chart(create_gauge_chart(current, max_hist), use_container_width=True)

with tab2:
    all_states_df = get_all_states_latest_unemployment()
    if all_states_df is not None:
        st.plotly_chart(create_choropleth_map(all_states_df), use_container_width=True)
    else:
        st.warning("Could not load map data.")

with tab3:
    chart_df = display_data_df.last('24M')
    metrics_for_chart1 = [m for m in ["Job Openings", "Unemployment Rate"] if m in chart_df.columns]
    if len(metrics_for_chart1) == 2:
        st.plotly_chart(create_time_series_chart(chart_df, st.session_state.selected_location, metrics_for_chart1), use_container_width=True)
    if 'Quits Rate' in chart_df.columns:
        st.plotly_chart(create_quits_rate_chart(chart_df, st.session_state.selected_location), use_container_width=True)

with tab4:
    st.subheader("Data Export")
    if display_data_df is not None and not display_data_df.empty:
        st.dataframe(display_data_df.tail(12).style.format("{:.2f}"))
        csv = display_data_df.to_csv().encode('utf-8')
        st.download_button("Download Full Data as CSV", data=csv, file_name=f"labor_pulse_{st.session_state.selected_location}.csv", mime='text/csv')
