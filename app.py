# Labor-Market Pulse: Professional Edition
# Enhanced version with improved readability and user experience

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
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    line-height: 1.2;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #6c757d;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Enhanced Cards */
.metric-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

/* Dark theme adjustments */
[data-theme="dark"] .metric-card {
    background: linear-gradient(145deg, #262730 0%, #1e1e1e 100%);
    border: 1px solid #404040;
}

[data-theme="dark"] .metric-card::before {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}

/* Info Cards */
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
}

.warning-card {
    background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    color: #2d3436;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(255, 234, 167, 0.3);
}

.error-card {
    background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(253, 121, 168, 0.3);
}

/* Sidebar Enhancements */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
}

[data-theme="dark"] .css-1d391kg {
    background: linear-gradient(180deg, #1e1e1e 0%, #262730 100%);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: 1px solid rgba(0,0,0,0.1);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Metric styling improvements */
.stMetric {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.9rem;
    color: #6c757d;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Loading animation */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .metric-card {
        padding: 1rem;
    }
}

/* Section dividers */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
    margin: 2rem 0;
    border: none;
}

/* Tooltip styling */
.tooltip {
    position: relative;
    cursor: help;
}

.tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: white;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    white-space: nowrap;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# --- BLS API Configuration ---
BLS_API_KEY = os.environ.get("BLS_API_KEY") or st.secrets.get("BLS_API_KEY")
if not BLS_API_KEY:
    st.markdown("""
    <div class="error-card">
        <h3>⚠️ API Key Required</h3>
        <p>A BLS API key is required to fetch labor market data. Please configure it in your Streamlit secrets.</p>
    </div>
    """, unsafe_allow_html=True)
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
    "Total Nonfarm": "000000", 
    "Construction": "200000", 
    "Manufacturing": "300000",
    "Trade, Transportation, and Utilities": "400000", 
    "Information": "510000",
    "Financial Activities": "520000", 
    "Professional and Business Services": "600000",
    "Education and Health Services": "620000", 
    "Leisure and Hospitality": "700000"
}

MSA_CODES = {
    "New York-Newark-Jersey City, NY-NJ-PA": "35620", 
    "Los Angeles-Long Beach-Anaheim, CA": "31080",
    "Chicago-Naperville-Elgin, IL-IN-WI": "16980", 
    "Dallas-Fort Worth-Arlington, TX": "19100",
    "Houston-The Woodlands-Sugar Land, TX": "26420", 
    "Washington-Arlington-Alexandria, DC-VA-MD-WV": "47900",
    "Miami-Fort Lauderdale-Pompano Beach, FL": "33100", 
    "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD": "37980",
    "Atlanta-Sandy Springs-Alpharetta, GA": "12060", 
    "Boston-Cambridge-Newton, MA-NH": "14460"
}

# --- Helper Functions ---
def show_loading_message(message="Loading data..."):
    """Display a loading message with spinner"""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: #6c757d;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def get_series_ids(loc_type, location, industry):
    """Get BLS series IDs based on location type and industry"""
    series = {}
    if loc_type == "U.S. Total":
        ind_code = INDUSTRY_CODES[industry]
        series["Job Openings"] = f"JTS{ind_code}000000000JOL"
        if industry == "Total Nonfarm":
            series["Unemployment Rate"] = "LNS14000000"
            series["Quits Rate"] = "JTS000000000000000QUR"
    elif loc_type == "State":
        fips = STATE_FIPS.get(location)
        if fips:
            series["Unemployment Rate"] = f"LASST{fips}0000000000003"
            if industry == "Total Nonfarm":
                series["Job Openings"] = f"JTS{fips}000000000JOL"
                series["Quits Rate"] = f"JTS{fips}000000000QUR"
    elif loc_type == "Metropolitan Area":
        msa_code = MSA_CODES.get(location)
        if msa_code:
            series["Unemployment Rate"] = f"LAUMT{msa_code}00000000003"
    return series

@st.cache_data(ttl=3600, show_spinner=False)
def get_bls_data(series_ids, years_to_fetch=5):
    """Fetch data from BLS API with improved error handling"""
    if not series_ids: 
        return None
    
    end_year = date.today().year
    start_year = end_year - years_to_fetch
    
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": list(series_ids.values()), 
        "startyear": str(start_year), 
        "endyear": str(end_year), 
        "registrationkey": BLS_API_KEY
    })
    
    try:
        response = requests.post(BLS_API_URL, data=data, headers=headers, timeout=20)
        response.raise_for_status()
        json_data = response.json()
        
        if json_data['status'] != 'REQUEST_SUCCEEDED': 
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
        
        if not all_series_data: 
            return None
            
        return pd.concat(all_series_data, axis=1).sort_index().ffill()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_states_latest_unemployment():
    """Fetch latest unemployment data for all states"""
    series_ids = {f"LASST{fips}0000000000003": name for name, fips in STATE_FIPS.items()}
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=12*30)).year
    
    headers = {'Content-type': 'application/json'}
    series_chunks = [list(series_ids.keys())[i:i + 50] for i in range(0, len(series_ids), 50)]
    latest_data = {}
    
    for chunk in series_chunks:
        data = json.dumps({
            "seriesid": chunk, 
            "startyear": str(start_year), 
            "endyear": str(end_year), 
            "registrationkey": BLS_API_KEY
        })
        
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
        except requests.exceptions.RequestException:
            continue
    
    if not latest_data: 
        return None
        
    df = pd.DataFrame(list(latest_data.items()), columns=['State', 'Unemployment Rate'])
    df['State_Abbr'] = df['State'].map(STATE_MAPPING)
    return df

# --- Enhanced Visualization Components ---
def create_choropleth_map(df):
    """Create an enhanced choropleth map"""
    fig = px.choropleth(
        df, 
        locations='State_Abbr', 
        locationmode="USA-states", 
        color='Unemployment Rate',
        color_continuous_scale="Viridis", 
        scope="usa", 
        hover_name='State',
        title="Latest Unemployment Rate by State", 
        labels={'Unemployment Rate': 'Rate (%)'},
        hover_data={'Unemployment Rate': ':.1f%'}
    )
    
    fig.update_layout(
        margin=dict(t=60, b=20, l=20, r=20), 
        title_x=0.5,
        title_font_size=16,
        title_font_family="Inter",
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        font_family="Inter"
    )
    
    return fig

def create_time_series_chart(df, location, metrics):
    """Create an enhanced time series chart"""
    fig = go.Figure()
    
    # Enhanced color palette
    colors = {
        'openings': '#667eea', 
        'unemployment': '#764ba2', 
        'grid': '#e9ecef'
    }
    
    if 'Job Openings' in metrics:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Job Openings'] / 1000, 
            name='Job Openings (K)', 
            mode='lines+markers',
            line=dict(color=colors['openings'], width=3),
            marker=dict(size=4),
            hovertemplate='<b>%{y:,.0f}K</b> openings<br>%{x}<extra></extra>'
        ))
    
    if 'Unemployment Rate' in metrics:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Unemployment Rate'], 
            name='Unemployment Rate (%)', 
            mode='lines+markers',
            line=dict(color=colors['unemployment'], width=3, dash='dash'),
            marker=dict(size=4),
            yaxis='y2',
            hovertemplate='<b>%{y:.1f}%</b> unemployment<br>%{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Labor Market Trends: {location}</b>', 
            font_size=18, 
            x=0.5,
            font_family="Inter"
        ),
        xaxis=dict(
            title_text=None, 
            showgrid=False,
            tickfont_family="Inter"
        ),
        yaxis=dict(
            title=dict(text='Job Openings (Thousands)', font_family="Inter"), 
            showgrid=True, 
            gridcolor=colors['grid'],
            tickfont_family="Inter"
        ),
        yaxis2=dict(
            title=dict(text='Unemployment Rate (%)', font_family="Inter"), 
            overlaying='y', 
            side='right', 
            showgrid=False,
            tickfont_family="Inter"
        ),
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            font_family="Inter"
        ),
        template='plotly_white', 
        height=400, 
        margin=dict(l=50, r=50, t=80, b=50), 
        hovermode='x unified',
        font_family="Inter"
    )
    
    return fig

def create_quits_rate_chart(df, location):
    """Create an enhanced quits rate chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Quits Rate'], 
        name='Quits Rate (%)', 
        mode='lines+markers',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=4),
        fill='tonexty',
        fillcolor='rgba(46, 204, 113, 0.1)',
        hovertemplate='<b>%{y:.1f}%</b> quits rate<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Quits Rate Trend: {location}</b>', 
            font_size=18, 
            x=0.5,
            font_family="Inter"
        ),
        xaxis=dict(
            title_text='Date', 
            showgrid=False,
            tickfont_family="Inter"
        ),
        yaxis=dict(
            title=dict(text='Quits Rate (%)', font_family="Inter"), 
            showgrid=True, 
            gridcolor='#e9ecef',
            tickfont_family="Inter"
        ),
        template='plotly_white', 
        height=350, 
        margin=dict(l=50, r=50, t=80, b=50), 
        hovermode='x unified',
        font_family="Inter"
    )
    
    return fig

def create_enhanced_sparkline(df, metric):
    """Create an enhanced sparkline chart"""
    recent_data = df[metric].iloc[-12:]
    
    spark = go.Figure(go.Scatter(
        x=recent_data.index, 
        y=recent_data.values, 
        mode='lines',
        line=dict(width=2, color='#667eea'),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    spark.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=60, 
        xaxis_visible=False, 
        yaxis_visible=False, 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return spark

def create_enhanced_gauge(current_value, historical_max, title):
    """Create an enhanced gauge chart"""
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'family': 'Inter'}},
        delta={'reference': historical_max * 0.5},
        gauge={
            'axis': {'range': [None, historical_max]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, historical_max * 0.3], 'color': "lightgray"},
                {'range': [historical_max * 0.3, historical_max * 0.7], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': historical_max * 0.9
            }
        }
    ))
    
    gauge.update_layout(
        height=280, 
        margin=dict(l=30, r=30, t=60, b=30),
        font_family="Inter"
    )
    
    return gauge

# --- State Management ---
def initialize_session_state():
    """Initialize session state variables"""
    if 'init' not in st.session_state:
        st.session_state.loc_type = "U.S. Total"
        st.session_state.selected_location = "U.S. Total"
        st.session_state.selected_industry = "Total Nonfarm"
        st.session_state.base_month = None
        st.session_state.init = True

def reset_to_defaults():
    """Reset all selections to default values"""
    st.session_state.loc_type = "U.S. Total"
    st.session_state.selected_location = "U.S. Total"
    st.session_state.selected_industry = "Total Nonfarm"
    st.session_state.base_month = None

initialize_session_state()

# --- Header Section ---
st.markdown('<h1 class="main-title">📊 Labor Market Pulse</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time insights into U.S. employment trends and workforce dynamics</p>', unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    # Get base data for date range
    with st.spinner("Loading date range..."):
        base_data_df = get_bls_data(get_series_ids("U.S. Total", "U.S. Total", "Total Nonfarm"))
    
    if base_data_df is not None:
        min_date, max_date = base_data_df.index.min(), base_data_df.index.max()
        
        if st.session_state.base_month is None:
            st.session_state.base_month = max_date.to_pydatetime()
        
        selected_date = st.slider(
            "📅 Select Base Month:", 
            min_value=min_date.to_pydatetime(), 
            max_value=max_date.to_pydatetime(), 
            value=st.session_state.base_month, 
            format="MMM YYYY",
            help="Choose the end date for your analysis"
        )
        st.session_state.base_month = selected_date

    st.markdown("#### 📍 Location Settings")
    st.radio(
        "Location Type:", 
        ["U.S. Total", "State", "Metropolitan Area"], 
        key='loc_type', 
        horizontal=True,
        help="Select the geographic scope for your analysis"
    )
    
    if st.session_state.loc_type == "State":
        st.selectbox(
            "Select State:", 
            sorted(STATE_FIPS.keys()), 
            key='selected_location',
            help="Choose a specific state for detailed analysis"
        )
    elif st.session_state.loc_type == "Metropolitan Area":
        st.selectbox(
            "Select Metro Area:", 
            sorted(MSA_CODES.keys()), 
            key='selected_location',
            help="Choose a metropolitan statistical area"
        )
    else:
        st.session_state.selected_location = "U.S. Total"
        st.selectbox(
            "Industry Sector:", 
            list(INDUSTRY_CODES.keys()), 
            key='selected_industry',
            help="Select an industry for focused analysis"
        )

    st.markdown("#### ⚙️ Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🔄 Refresh', use_container_width=True, help="Clear cache and reload all data"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button('↩️ Reset', use_container_width=True, help="Reset all settings to defaults"):
            reset_to_defaults()
            st.rerun()

    # Information section
    with st.expander("📋 About This Dashboard"):
        st.markdown("""
        **Data Source:** U.S. Bureau of Labor Statistics (BLS)
        
        **Key Metrics:**
        - **Unemployment Rate:** Percentage of labor force unemployed
        - **Job Openings:** Total available positions (JOLTS data)
        - **Quits Rate:** Voluntary job separations as % of employment
        
        **Design Principles:**
        - 📊 Visual integrity with minimal chart junk
        - 🎨 Colorblind-safe palettes
        - 📱 Responsive design for all devices
        
        **Update Frequency:** Data refreshes hourly
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <h4>💡 Pro Tip</h4>
        <p>Use the time slider to compare current metrics with historical periods. Hover over charts for detailed insights!</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Dashboard Content ---
# Dynamic location title
loc_title = st.session_state.selected_location
if st.session_state.loc_type == "U.S. Total" and st.session_state.selected_industry != "Total Nonfarm":
    loc_title += f" - {st.session_state.selected_industry}"

st.markdown(f"## 📈 Analytics Dashboard: {loc_title}")

# Get data for selected location and settings
series_ids = get_series_ids(
    st.session_state.loc_type, 
    st.session_state.selected_location, 
    st.session_state.selected_industry if st.session_state.loc_type == "U.S. Total" else "Total Nonfarm"
)

# Show loading spinner while fetching data
with st.spinner("Fetching labor market data..."):
    full_data_df = get_bls_data(series_ids)

if full_data_df is None:
    st.markdown("""
    <div class="error-card">
        <h3>❌ Data Unavailable</h3>
        <p>Could not retrieve data for the selected filters. This might be due to:</p>
        <ul>
            <li>API rate limits</li>
            <li>Data not available for this location/industry combination</li>
            <li>Temporary service issues</li>
        </ul>
        <p>Please try a different selection or refresh the page.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Filter data based on selected date
display_data_df = full_data_df[full_data_df.index <= pd.to_datetime(st.session_state.base_month)]

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ State Map", "📈 Trends", "📋 Data Export"])

# --- Tab 1: Overview ---
with tab1:
    st.markdown("### 🎯 Key Performance Indicators")
    
    if display_data_df is not None and len(display_data_df) > 1:
        latest_date_str = display_data_df.index[-1].strftime('%b %Y')
        previous_date_str = display_data_df.index[-2].strftime('%B %Y')
        
        # Create metrics columns
        metrics_to_show = ["Unemployment Rate", "Job Openings", "Quits Rate"]
        available_metrics = [m for m in metrics_to_show if m in display_data_df.columns]
        
        if available_metrics:
            cols = st.columns(len(available_metrics))
            
            for i, metric in enumerate(available_metrics):
                with cols[i]:
                    # Get values
                    latest_val = display_data_df[metric].iloc[-1]
                    delta = latest_val - display_data_df[metric].iloc[-2] if len(display_data_df) > 1 else 0
                    
                    # Create custom metric card
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    
                    # Metric-specific formatting and icons
                    if metric == "Unemployment Rate":
                        icon = "📉" if delta < 0 else "📈"
                        formatted_value = f"{latest_val:.1f}%"
                        formatted_delta = f"{delta:+.2f}%"
                        delta_color = "🟢" if delta < 0 else "🔴"
                        help_text = f"Change from {previous_date_str}. Lower unemployment is generally better for the economy."
                        
                    elif metric == "Job Openings":
                        icon = "💼"
                        if latest_val >= 1e6:
                            formatted_value = f"{latest_val/1e6:.1f}M"
                            formatted_delta = f"{delta/1e6:+.1f}M"
                        else:
                            formatted_value = f"{latest_val/1e3:,.0f}K"
                            formatted_delta = f"{delta/1e3:+.1f}K"
                        delta_color = "🟢" if delta > 0 else "🔴"
                        help_text = f"Change from {previous_date_str}. More openings indicate a tighter labor market."
                        
                    elif metric == "Quits Rate":
                        icon = "🚪"
                        formatted_value = f"{latest_val:.1f}%"
                        formatted_delta = f"{delta:+.2f}%"
                        delta_color = "🟢" if delta > 0 else "🔴"
                        help_text = f"Change from {previous_date_str}. Higher quits rate indicates worker confidence."
                    
                    # Display metric
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                        <div class="metric-label">{metric}</div>
                        <div class="metric-value">{formatted_value}</div>
                        <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 1rem;">
                            {delta_color} {formatted_delta} vs. {previous_date_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add sparkline
                    if len(display_data_df) >= 12:
                        st.plotly_chart(
                            create_enhanced_sparkline(display_data_df, metric), 
                            use_container_width=True, 
                            config={'displayModeBar': False}
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Add section divider
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Unemployment Rate Gauge
        if "Unemployment Rate" in display_data_df.columns:
            st.markdown("### 🎯 Unemployment Rate Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                current_rate = display_data_df['Unemployment Rate'].iloc[-1]
                historical_max = display_data_df['Unemployment Rate'].max()
                historical_min = display_data_df['Unemployment Rate'].min()
                
                gauge_fig = create_enhanced_gauge(
                    current_rate, 
                    historical_max, 
                    f"Current: {current_rate:.1f}% vs Historical Range"
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>📊 Statistical Summary</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Statistical metrics
                st.metric(
                    "Historical High", 
                    f"{historical_max:.1f}%", 
                    help="Highest unemployment rate in the dataset"
                )
                st.metric(
                    "Historical Low", 
                    f"{historical_min:.1f}%", 
                    help="Lowest unemployment rate in the dataset"
                )
                st.metric(
                    "Average", 
                    f"{display_data_df['Unemployment Rate'].mean():.1f}%", 
                    help="Average unemployment rate over the period"
                )
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Insufficient Data</h3>
            <p>Not enough historical data available to display KPIs. Please select a different time period or location.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 2: State Map ---
with tab2:
    st.markdown("### 🗺️ National Unemployment Overview")
    
    with st.spinner("Loading state-level data..."):
        all_states_df = get_all_states_latest_unemployment()
    
    if all_states_df is not None:
        # Add summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "National Average", 
                f"{all_states_df['Unemployment Rate'].mean():.1f}%",
                help="Average unemployment rate across all states"
            )
        
        with col2:
            highest_state = all_states_df.loc[all_states_df['Unemployment Rate'].idxmax()]
            st.metric(
                "Highest Rate", 
                f"{highest_state['Unemployment Rate']:.1f}%",
                delta=f"{highest_state['State']}",
                help="State with the highest unemployment rate"
            )
        
        with col3:
            lowest_state = all_states_df.loc[all_states_df['Unemployment Rate'].idxmin()]
            st.metric(
                "Lowest Rate", 
                f"{lowest_state['Unemployment Rate']:.1f}%",
                delta=f"{lowest_state['State']}",
                help="State with the lowest unemployment rate"
            )
        
        with col4:
            range_val = all_states_df['Unemployment Rate'].max() - all_states_df['Unemployment Rate'].min()
            st.metric(
                "Range", 
                f"{range_val:.1f}pp",
                help="Difference between highest and lowest state rates"
            )
        
        # Choropleth map
        st.plotly_chart(create_choropleth_map(all_states_df), use_container_width=True)
        
        # Top/Bottom states
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Highest Unemployment Rates")
            top_5 = all_states_df.nlargest(5, 'Unemployment Rate')[['State', 'Unemployment Rate']]
            for idx, row in top_5.iterrows():
                st.markdown(f"**{row['State']}:** {row['Unemployment Rate']:.1f}%")
        
        with col2:
            st.markdown("#### 🟢 Lowest Unemployment Rates")
            bottom_5 = all_states_df.nsmallest(5, 'Unemployment Rate')[['State', 'Unemployment Rate']]
            for idx, row in bottom_5.iterrows():
                st.markdown(f"**{row['State']}:** {row['Unemployment Rate']:.1f}%")
    else:
        st.markdown("""
        <div class="error-card">
            <h3>❌ Map Data Unavailable</h3>
            <p>Could not load state-level unemployment data. This may be due to API limits or service issues.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 3: Historical Trends ---
with tab3:
    st.markdown("### 📈 Historical Trends Analysis")
    
    if display_data_df is not None and not display_data_df.empty:
        # Time range selector
        col1, col2 = st.columns(2)
        
        with col1:
            time_range = st.selectbox(
                "Select Time Range:",
                ["Last 12 months", "Last 24 months", "Last 36 months", "All available data"],
                index=1,
                help="Choose the time period for trend analysis"
            )
        
        with col2:
            chart_style = st.selectbox(
                "Chart Style:",
                ["Lines + Markers", "Lines Only", "Area Chart"],
                help="Select visualization style"
            )
        
        # Filter data based on time range
        if time_range == "Last 12 months":
            chart_df = display_data_df.last('12M')
        elif time_range == "Last 24 months":
            chart_df = display_data_df.last('24M')
        elif time_range == "Last 36 months":
            chart_df = display_data_df.last('36M')
        else:
            chart_df = display_data_df
        
        # Main trends chart
        metrics_for_chart = [m for m in ["Job Openings", "Unemployment Rate"] if m in chart_df.columns]
        
        if len(metrics_for_chart) >= 1:
            st.plotly_chart(
                create_time_series_chart(chart_df, st.session_state.selected_location, metrics_for_chart), 
                use_container_width=True
            )
        
        # Quits rate chart (if available)
        if 'Quits Rate' in chart_df.columns:
            st.plotly_chart(
                create_quits_rate_chart(chart_df, st.session_state.selected_location), 
                use_container_width=True
            )
        
        # Trend analysis insights
        st.markdown("#### 🔍 Trend Insights")
        
        insights_cols = st.columns(2)
        
        with insights_cols[0]:
            if 'Unemployment Rate' in chart_df.columns and len(chart_df) > 1:
                unemployment_trend = chart_df['Unemployment Rate'].iloc[-1] - chart_df['Unemployment Rate'].iloc[0]
                trend_direction = "📈 Rising" if unemployment_trend > 0 else "📉 Falling"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h5>Unemployment Trend</h5>
                    <p><strong>{trend_direction}</strong> by {abs(unemployment_trend):.1f} percentage points over the selected period</p>
                </div>
                """, unsafe_allow_html=True)
        
        with insights_cols[1]:
            if 'Job Openings' in chart_df.columns and len(chart_df) > 1:
                openings_trend = chart_df['Job Openings'].iloc[-1] - chart_df['Job Openings'].iloc[0]
                trend_direction = "📈 Increasing" if openings_trend > 0 else "📉 Decreasing"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h5>Job Openings Trend</h5>
                    <p><strong>{trend_direction}</strong> by {abs(openings_trend/1000):,.0f}K positions over the selected period</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ No Trend Data</h3>
            <p>No historical data available for trend analysis. Please select a different location or time period.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 4: Data Export ---
with tab4:
    st.markdown("### 📋 Data Export & Summary")
    
    if display_data_df is not None and not display_data_df.empty:
        # Data summary
        st.markdown("#### 📊 Dataset Summary")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Data Points", len(display_data_df))
        
        with summary_cols[1]:
            st.metric("Date Range", f"{len(display_data_df)} months")
        
        with summary_cols[2]:
            st.metric("Metrics Available", len(display_data_df.columns))
        
        with summary_cols[3]:
            latest_update = display_data_df.index[-1].strftime('%b %Y')
            st.metric("Latest Data", latest_update)
        
        # Data preview
        st.markdown("#### 👀 Data Preview")
        
        # Show last 10 rows
        preview_df = display_data_df.tail(10).copy()
        preview_df.index = preview_df.index.strftime('%Y-%m')
        
        # Format columns for display
        for col in preview_df.columns:
            if 'Rate' in col:
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:.1f}%")
            elif 'Openings' in col:
                preview_df[col] = preview_df[col].apply(lambda x: f"{x/1000:,.0f}K")
        
        st.dataframe(preview_df, use_container_width=True)
        
        # Download options
        st.markdown("#### 💾 Download Options")
        
        download_cols = st.columns(3)
        
        with download_cols[0]:
            # CSV download
            csv_data = display_data_df.to_csv().encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"labor_market_data_{st.session_state.selected_location.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                use_container_width=True,
                help="Download the raw data in CSV format"
            )
        
        with download_cols[1]:
            # JSON download
            json_data = display_data_df.to_json(orient='index', date_format='iso').encode('utf-8')
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"labor_market_data_{st.session_state.selected_location.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.json",
                mime='application/json',
                use_container_width=True,
                help="Download the data in JSON format"
            )
        
        with download_cols[2]:
            # Excel download would require additional libraries
            st.markdown("""
            <div style="padding: 0.5rem; text-align: center; color: #6c757d; border: 1px dashed #dee2e6; border-radius: 4px;">
                📊 Excel format<br>
                <small>Available in full version</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Data source information
        st.markdown("#### ℹ️ Data Source Information")
        st.markdown("""
        <div class="info-card">
            <h5>U.S. Bureau of Labor Statistics (BLS)</h5>
            <ul>
                <li><strong>Unemployment Rate:</strong> Local Area Unemployment Statistics (LAUS)</li>
                <li><strong>Job Openings:</strong> Job Openings and Labor Turnover Survey (JOLTS)</li>
                <li><strong>Quits Rate:</strong> Job Openings and Labor Turnover Survey (JOLTS)</li>
            </ul>
            <p><strong>Update Frequency:</strong> Monthly data, typically released with a 1-2 month lag</p>
            <p><strong>Geographic Coverage:</strong> National, state, and metropolitan statistical areas</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ No Data to Export</h3>
            <p>No data available for the current selection. Please adjust your filters and try again.</p>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d;">
    <p>📊 <strong>Labor Market Pulse</strong> - Professional Edition</p>
    <p>Built with ❤️ using Streamlit • Data from U.S. Bureau of Labor Statistics</p>
    <p><small>Last updated: {}</small></p>
</div>
""".format(date.today().strftime('%B %d, %Y')), unsafe_allow_html=True)
