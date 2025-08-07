import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta, datetime
import os
import json
import sqlite3
import time
import logging
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from typing import List, Dict, Any, Optional, Callable
from statsmodels.tsa.seasonal import seasonal_decompose
import contextlib

# --- 1. Configuration Management ---
@dataclass
class AppConfig:
    """Manages application-wide settings and configurations."""
    APP_TITLE: str = "Labor-Market Pulse"
    PAGE_ICON: str = "📊"
    API_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    API_TIMEOUT: int = 20
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_DB_PATH: str = "bls_api_cache.db"
    LOG_FILE: str = "app.log"
    LOG_LEVEL: int = logging.INFO
    YEARS_OF_DATA: int = 5

    # Visualization settings
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#0D6EFD',
        'secondary': '#6C757D',
        'success': '#198754',
        'grid': '#ddd',
        'background_light': '#FFFFFF',
        'background_dark': '#262730',
        'card_border_light': '#E0E0E0',
        'card_border_dark': '#444'
    })

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.API_TIMEOUT <= 0:
            raise ValueError("API_TIMEOUT must be a positive integer.")
        if self.CACHE_TTL_SECONDS < 0:
            raise ValueError("CACHE_TTL_SECONDS cannot be negative.")
        logging.basicConfig(level=self.LOG_LEVEL,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(self.LOG_FILE), logging.StreamHandler()])

@dataclass
class SecureConfig:
    """Manages sensitive configuration like API keys."""
    bls_api_key: Optional[str] = None

    def __post_init__(self):
        """Load and validate the API key."""
        self.bls_api_key = os.environ.get("BLS_API_KEY") or st.secrets.get("BLS_API_KEY")
        if not self.bls_api_key or len(self.bls_api_key) < 30:
            st.error("A valid BLS_API_KEY is required. Please set it as a Streamlit secret.")
            logging.critical("BLS API Key is missing or invalid.")
            st.stop()

    @property
    def masked_key(self) -> str:
        """Returns a masked version of the API key for logging."""
        return f"{self.bls_api_key[:4]}...{self.bls_api_key[-4:]}" if self.bls_api_key else "N/A"

# --- 2. Performance Monitoring & Error Handling ---
class MetricsCollector:
    """Singleton class to collect performance metrics."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.api_calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.response_times = []

    def record_api_call(self, response_time: float):
        self.api_calls += 1
        self.response_times.append(response_time)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_error(self):
        self.errors += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "Total API Calls": self.api_calls,
            "Cache Hits": self.cache_hits,
            "Total Errors": self.errors,
            "Average Response Time (s)": round(sum(self.response_times) / len(self.response_times), 2) if self.response_times else 0
        }

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def handle_errors(default_return: Any = None, show_error: bool = True) -> Callable:
    """Decorator for robust error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in '{func.__name__}': {e}", exc_info=True)
                MetricsCollector().record_error()
                if show_error:
                    st.error(f"An unexpected error occurred in {func.__name__}. Please check the logs.")
                return default_return
        return wrapper
    return decorator

# --- 3. Data Layer ---
class DataMappings:
    """Manages static data mappings."""
    STATE_MAPPING: Dict[str, str] = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
    STATE_FIPS: Dict[str, str] = {'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35', 'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '52', 'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'}
    INDUSTRY_CODES: Dict[str, Dict[str, str]] = {"Total Nonfarm": {"JOLTS": "000000", "CPS_UR": "LNS14000000"}, "Construction": {"JOLTS": "200000", "CPS_UR": "LNU04032231"}, "Manufacturing": {"JOLTS": "300000", "CPS_UR": "LNU04032232"}, "Trade, Transportation, and Utilities": {"JOLTS": "400000", "CPS_UR": "LNU04032235"}, "Information": {"JOLTS": "510000", "CPS_UR": "LNU04032240"}, "Financial Activities": {"JOLTS": "520000", "CPS_UR": "LNU04032241"}, "Professional and Business Services": {"JOLTS": "600000", "CPS_UR": "LNU04032242"}, "Education and Health Services": {"JOLTS": "620000", "CPS_UR": "LNU04032243"}, "Leisure and Hospitality": {"JOLTS": "700000", "CPS_UR": "LNU04032244"}}

    @staticmethod
    @lru_cache(maxsize=128)
    def get_series_ids(loc_type: str, location: str, industry: str) -> Dict[str, str]:
        series = {}
        if loc_type == "U.S. Total":
            series["Job Openings"] = f"JTS{DataMappings.INDUSTRY_CODES[industry]['JOLTS']}000000000JOL"
            series["Unemployment Rate"] = DataMappings.INDUSTRY_CODES[industry]['CPS_UR']
            if industry == "Total Nonfarm":
                series["Quits Rate"] = "JTS000000000000000QUR"
        elif loc_type == "State":
            fips = DataMappings.STATE_FIPS.get(location)
            if fips:
                series["Unemployment Rate"] = f"LASST{fips}0000000000003"
                series["Job Openings"] = f"JTS{fips}000000000JOL"
                series["Quits Rate"] = f"JTS{fips}000000000QUR"
        return series

class LocalCache:
    """Manages a local SQLite cache for API responses."""
    def __init__(self, config: AppConfig):
        self.db_path = config.CACHE_DB_PATH
        self.ttl = config.CACHE_TTL_SECONDS
        self._create_table()

    def _create_table(self):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
            """)
            conn.commit()

    @contextlib.contextmanager
    def connect(self):
        """Provides a transactional scope around a series of operations."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get(self, key: str) -> Optional[Any]:
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response, timestamp FROM api_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                response_str, timestamp = row
                if time.time() - timestamp < self.ttl:
                    MetricsCollector().record_cache_hit()
                    logging.info(f"Cache hit for key: {key}")
                    return json.loads(response_str)
                else:
                    logging.info(f"Cache expired for key: {key}")
        return None

    def set(self, key: str, value: Any):
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("REPLACE INTO api_cache (key, response, timestamp) VALUES (?, ?, ?)",
                           (key, json.dumps(value), time.time()))
            conn.commit()
            logging.info(f"Cache set for key: {key}")

    def cleanup(self):
        """Removes expired entries from the cache."""
        with self.connect() as conn:
            cursor = conn.cursor()
            expiration_time = time.time() - self.ttl
            cursor.execute("DELETE FROM api_cache WHERE timestamp < ?", (expiration_time,))
            conn.commit()
            logging.info("Cache cleanup performed.")

class BLSAPIClient:
    """Client for interacting with the BLS API, with caching and retries."""
    def __init__(self, config: AppConfig, secure_config: SecureConfig, cache: LocalCache):
        self.config = config
        self.secure_config = secure_config
        self.cache = cache
        self.session = requests.Session()
        self.metrics = MetricsCollector()

    @handle_errors(default_return=None)
    @monitor_performance
    def get_data(self, series_ids: Dict[str, str], years_to_fetch: int) -> Optional[pd.DataFrame]:
        if not series_ids: return None
        
        cache_key = f"{'_'.join(sorted(series_ids.values()))}_{years_to_fetch}"
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return self._process_response(cached_response, series_ids)

        end_year, start_year = date.today().year, date.today().year - years_to_fetch
        data = json.dumps({
            "seriesid": list(series_ids.values()),
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": self.secure_config.bls_api_key
        })

        for i in range(3): # Retry logic
            try:
                start_time = time.time()
                response = self.session.post(self.config.API_URL, data=data, headers={'Content-type': 'application/json'}, timeout=self.config.API_TIMEOUT)
                self.metrics.record_api_call(time.time() - start_time)
                response.raise_for_status()
                json_data = response.json()

                if json_data.get('status') == 'REQUEST_SUCCEEDED':
                    self.cache.set(cache_key, json_data)
                    return self._process_response(json_data, series_ids)
                else:
                    logging.error(f"BLS API Error: {json_data.get('message')}")
                    return None
            except requests.exceptions.RequestException as e:
                logging.warning(f"API call failed (attempt {i+1}/3): {e}")
                time.sleep(1 * (2 ** i)) # Exponential backoff
        
        logging.error("API call failed after multiple retries.")
        return None

    def _process_response(self, json_data: Dict[str, Any], series_ids: Dict[str, str]) -> Optional[pd.DataFrame]:
        all_series_data = []
        for series_name, series_id in series_ids.items():
            for result_series in json_data['Results']['series']:
                if result_series['seriesID'] == series_id and result_series['data']:
                    df = pd.DataFrame(result_series['data'])
                    df['date'] = pd.to_datetime(df['year'] + '-' + df['periodName'])
                    df = df.set_index('date')[['value']].rename(columns={'value': series_name})
                    df[series_name] = pd.to_numeric(df[series_name], errors='coerce')
                    all_series_data.append(df)
                    break
        if not all_series_data: return None
        return pd.concat(all_series_data, axis=1).sort_index().ffill()
    
    @st.cache_data(ttl=3600)
    def get_all_states_latest_unemployment(_self): # Use _self to avoid conflict with instance self
        series_ids = {f"LASST{fips}0000000000003": name for name, fips in DataMappings.STATE_FIPS.items()}
        end_year, start_year = date.today().year, (date.today() - timedelta(days=12*30)).year
        headers = {'Content-type': 'application/json'}
        series_chunks = [list(series_ids.keys())[i:i + 50] for i in range(0, len(series_ids), 50)]
        latest_data = {}
        for chunk in series_chunks:
            data = json.dumps({"seriesid": chunk, "startyear": str(start_year), "endyear": str(end_year), "registrationkey": _self.secure_config.bls_api_key})
            try:
                response = _self.session.post(_self.config.API_URL, data=data, headers=headers, timeout=20)
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
        df['State_Abbr'] = df['State'].map(DataMappings.STATE_MAPPING)
        return df

# --- 4. Data Validation ---
class DataValidator:
    """Performs validation checks on DataFrames."""
    @staticmethod
    def validate_structure(df: pd.DataFrame, expected_cols: List[str]) -> bool:
        if not isinstance(df, pd.DataFrame):
            logging.error("Validation failed: Input is not a DataFrame.")
            return False
        if not all(col in df.columns for col in expected_cols):
            logging.error(f"Validation failed: Missing one of expected columns {expected_cols}.")
            return False
        return True

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include='number').columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

# --- 5. Session State Management ---
class SessionStateManager:
    """Manages the Streamlit session state."""
    DEFAULTS = {
        'loc_type': "U.S. Total",
        'selected_location': "U.S. Total",
        'selected_industry': "Total Nonfarm",
        'base_month': None,
        'last_updated': None,
        'init': True
    }
    def __init__(self):
        self.initialize()

    def initialize(self):
        for key, value in self.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def reset(self):
        st.cache_data.clear()
        for key, value in self.DEFAULTS.items():
            st.session_state[key] = value

    def get(self, key: str) -> Any:
        return st.session_state.get(key)

    def set(self, key: str, value: Any):
        st.session_state[key] = value

# --- 6. Visualization Factory ---
class ChartFactory:
    """Creates all Plotly visualizations for the dashboard."""
    def __init__(self, config: AppConfig):
        self.config = config

    def create_choropleth_map(self, df: pd.DataFrame) -> go.Figure:
        fig = px.choropleth(df, locations='State_Abbr', locationmode="USA-states", color='Unemployment Rate',
                            color_continuous_scale="Blues", scope="usa", hover_name='State',
                            title="Latest Unemployment Rate by State", labels={'Unemployment Rate': 'Rate (%)'})
        fig.update_layout(margin=dict(t=40,b=0,l=0,r=0), title_x=0.5, geo=dict(bgcolor='rgba(0,0,0,0)'))
        return fig

    def create_time_series_chart(self, df: pd.DataFrame, title: str, metrics: List[str], industry: str) -> go.Figure:
        fig = go.Figure()
        colors = self.config.COLORS
        industry_prefix = "" if industry == "Total Nonfarm" else f"{industry} "
        if 'Job Openings' in metrics:
            fig.add_trace(go.Scatter(x=df.index, y=df['Job Openings'] / 1000, name=f"{industry_prefix}Openings (K)", mode='lines', line=dict(color=colors['primary'], width=2.5)))
        if 'Unemployment Rate' in metrics:
            fig.add_trace(go.Scatter(x=df.index, y=df['Unemployment Rate'], name=f"{industry_prefix}Unemp. Rate (%)", mode='lines', line=dict(color=colors['secondary'], width=2.5, dash='dash'), yaxis='y2'))
        fig.update_layout(title=dict(text=f'<b>{title}</b>', font_size=16, x=0.5),
                          xaxis=dict(title_text=None, showgrid=False),
                          yaxis=dict(title=dict(text='Job Openings (K)'), showgrid=True, gridcolor=colors['grid']),
                          yaxis2=dict(title=dict(text='Unemployment (%)'), overlaying='y', side='right', showgrid=False),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                          template='plotly_white', height=350, margin=dict(l=50, r=50, t=50, b=20), hovermode='x unified')
        return fig
    
    def create_single_metric_chart(self, df: pd.DataFrame, title: str, metric_name: str, color: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[metric_name], name=metric_name, mode='lines', line=dict(color=color, width=2.5)))
        fig.update_layout(title=dict(text=f'<b>{title}</b>', font_size=16, x=0.5),
                          xaxis=dict(title_text='Date', showgrid=False),
                          yaxis=dict(title=dict(text=metric_name), showgrid=True, gridcolor='#ddd'),
                          template='plotly_white', height=350, margin=dict(l=50, r=50, t=50, b=50), hovermode='x unified')
        return fig

    def create_quits_rate_chart(self, df: pd.DataFrame, location: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Quits Rate'], name='Quits Rate (%)', mode='lines', line=dict(color=self.config.COLORS['success'], width=2.5)))
        fig.update_layout(title=dict(text=f'<b>Quits Rate Trend for {location}</b>', font_size=16, x=0.5),
                          xaxis=dict(title_text='Date', showgrid=False),
                          yaxis=dict(title=dict(text='Quits Rate (%)'), showgrid=True, gridcolor='#ddd'),
                          template='plotly_white', height=300, margin=dict(l=50, r=50, t=50, b=50), hovermode='x unified')
        return fig

    def create_sparkline(self, df: pd.DataFrame, metric: str) -> go.Figure:
        spark = go.Figure(go.Scatter(x=df.index[-12:], y=df[metric].iloc[-12:], mode='lines', line=dict(width=2)))
        spark.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=60, xaxis_visible=False, yaxis_visible=False, plot_bgcolor='rgba(0,0,0,0)')
        return spark

    def create_gauge_chart(self, current: float, max_hist: float) -> go.Figure:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=current,
            gauge={'axis':{'range':[0, max_hist]}, 'bar':{'color':self.config.COLORS['primary']}},
            title={'text':'Current Rate vs. Historical High'}
        ))
        gauge.update_layout(height=250, margin=dict(l=30,r=30,t=60,b=30))
        return gauge

# --- 7. Main Application Class ---
class LaborMarketApp:
    def __init__(self):
        self.config = AppConfig()
        self.secure_config = SecureConfig()
        self.cache = LocalCache(self.config)
        self.api_client = BLSAPIClient(self.config, self.secure_config, self.cache)
        self.validator = DataValidator()
        self.chart_factory = ChartFactory(self.config)
        self.state = SessionStateManager()
        self.metrics = MetricsCollector()

    def _render_sidebar(self):
        with st.sidebar:
            st.header("Controls & Information")
            base_data_df = self.api_client.get_data(DataMappings.get_series_ids("U.S. Total", "U.S. Total", "Total Nonfarm"), self.config.YEARS_OF_DATA)
            if base_data_df is not None:
                min_date, max_date = base_data_df.index.min(), base_data_df.index.max()
                if self.state.get('base_month') is None:
                    self.state.set('base_month', max_date.to_pydatetime())
                
                selected_date = st.slider("Select Base Month:", min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(), value=self.state.get('base_month'), format="MMM YYYY")
                self.state.set('base_month', selected_date)

            self.state.set('loc_type', st.radio("Location Type:", ["U.S. Total", "State"], index=["U.S. Total", "State"].index(self.state.get('loc_type')), horizontal=True))
            
            if self.state.get('loc_type') == "State":
                self.state.set('selected_location', st.selectbox("State:", sorted(DataMappings.STATE_FIPS.keys()), index=sorted(DataMappings.STATE_FIPS.keys()).index(self.state.get('selected_location')) if self.state.get('selected_location') in DataMappings.STATE_FIPS else 0))
                self.state.set('selected_industry', "Total Nonfarm")
            else: # US Total
                self.state.set('selected_location', "U.S. Total")
                self.state.set('selected_industry', st.selectbox("Industry:", list(DataMappings.INDUSTRY_CODES.keys()), index=list(DataMappings.INDUSTRY_CODES.keys()).index(self.state.get('selected_industry'))))

            col1, col2 = st.columns(2)
            if col1.button('Refresh All Data', use_container_width=True):
                self.cache.cleanup()
                st.cache_data.clear()
                self.state.set('last_updated', pd.Timestamp.now(tz="UTC"))
                st.rerun()
            col2.button('Reset All', on_click=self.state.reset, use_container_width=True)
            with st.expander("Design & Accessibility Notes"):
                st.markdown("- **Data Adjustment:** Most data is seasonally adjusted. National industry-level unemployment rates are not seasonally adjusted.\n- **Visual Integrity:** Minimize non-data ink.\n- **Color Choice:** Sequential, colorblind-safe palettes.")
            st.info("Data Source: U.S. Bureau of Labor Statistics (BLS)")
            if self.state.get('last_updated'):
                st.caption(f"Data last refreshed: {self.state.get('last_updated').strftime('%Y-%m-%d %H:%M:%S %Z')}")

    def _render_overview_tab(self, df: pd.DataFrame):
        st.subheader("Key Performance Indicators")
        if df.empty or len(df) < 2:
            st.warning("Not enough data to display KPIs for the selected period.")
            return

        latest_date_str = df.index[-1].strftime('%b %Y')
        previous_date_str = df.index[-2].strftime('%B %Y')
        
        metrics = [m for m in ["Unemployment Rate", "Job Openings", "Quits Rate"] if m in df.columns and not df[m].dropna().empty]
        if not metrics:
            st.warning("No valid KPI data found.")
            return

        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            with cols[i]:
                latest_val = df[metric].iloc[-1]
                delta = latest_val - df[metric].iloc[-2]
                
                value_str = f"{latest_val/1e6:.2f}M" if metric == "Job Openings" and latest_val >= 1e6 else f"{latest_val/1e3:,.0f}K" if metric == "Job Openings" else f"{latest_val:.1f}%"
                delta_str = f"{delta/1e3:+.1f}K" if metric == "Job Openings" else f"{delta:+.2f}%"
                
                st.metric(label=f"{metric} ({latest_date_str})", value=value_str, delta=f"{delta_str} vs {previous_date_str}", delta_color="inverse" if metric == "Unemployment Rate" else "normal")
                st.plotly_chart(self.chart_factory.create_sparkline(df, metric), use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")
        if "Unemployment Rate" in df.columns:
            st.plotly_chart(self.chart_factory.create_gauge_chart(df['Unemployment Rate'].iloc[-1], df['Unemployment Rate'].max()), use_container_width=True)

    def _render_trends_tab(self, df: pd.DataFrame, title: str, industry: str):
        metrics_for_chart = [m for m in ["Job Openings", "Unemployment Rate"] if m in df.columns and not df[m].dropna().empty]
        if len(metrics_for_chart) == 2:
            st.plotly_chart(self.chart_factory.create_time_series_chart(df, title, metrics_for_chart, industry), use_container_width=True)
        elif len(metrics_for_chart) == 1:
            metric = metrics_for_chart[0]
            color = self.config.COLORS['primary'] if metric == "Job Openings" else self.config.COLORS['secondary']
            
            if self.state.get('loc_type') in ["State"]:
                 st.info(f"Only {metric} data is available for this view.")
            else:
                 st.info(f"Only {metric} data is available for this view. This may be due to reporting lags for other metrics. Try selecting an earlier Base Month.")
            
            st.plotly_chart(self.chart_factory.create_single_metric_chart(df, title, metric, color), use_container_width=True)
        
        if 'Quits Rate' in df.columns and not df['Quits Rate'].dropna().empty:
            st.plotly_chart(self.chart_factory.create_quits_rate_chart(df, title), use_container_width=True)

    def run(self):
        st.markdown('<h1 class="main-title">Labor Market Pulse</h1>', unsafe_allow_html=True)
        self._render_sidebar()

        loc_title = self.state.get('selected_location')
        if self.state.get('loc_type') == "U.S. Total" and self.state.get('selected_industry') != "Total Nonfarm":
            loc_title += f" ({self.state.get('selected_industry')})"
        elif self.state.get('loc_type') != "U.S. Total":
            loc_title += " (Total Nonfarm)"
        st.header(f"Dashboard for {loc_title}")

        series_ids = DataMappings.get_series_ids(self.state.get('loc_type'), self.state.get('selected_location'), self.state.get('selected_industry'))
        
        with st.spinner("Fetching data from BLS API..."):
            full_data_df = self.api_client.get_data(series_ids, self.config.YEARS_OF_DATA)

        if full_data_df is None:
            st.error("Could not retrieve data for the selected filters. Please try a different selection.")
            st.stop()
        
        if self.state.get('last_updated') is None:
            self.state.set('last_updated', pd.Timestamp.now(tz="UTC"))

        display_data_df = full_data_df[full_data_df.index <= pd.to_datetime(self.state.get('base_month'))]

        if not display_data_df.empty:
            selected_month_ts = pd.to_datetime(self.state.get('base_month'))
            latest_available_ts = display_data_df.index[-1]
            if selected_month_ts.year > latest_available_ts.year or selected_month_ts.month > latest_available_ts.month:
                st.warning(f"Data for {selected_month_ts.strftime('%B %Y')} is not yet available. Showing data as of {latest_available_ts.strftime('%B %Y')}.")
            else:
                st.markdown(f"<p class='sub-header'>Data as of {latest_available_ts.strftime('%B %Y')}</p>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ State Map", "📈 Historical Trends", "📋 Data Export"])

        with tab1:
            self._render_overview_tab(display_data_df)
        with tab2:
            if self.state.get('loc_type') == "U.S. Total" and self.state.get('selected_industry') != "Total Nonfarm":
                st.info("Please note: The map below displays the overall 'Total Nonfarm' unemployment rate for each state, as industry-specific unemployment data is not available at the state level.")
            all_states_df = self.api_client.get_all_states_latest_unemployment()
            if all_states_df is not None:
                st.plotly_chart(self.chart_factory.create_choropleth_map(all_states_df), use_container_width=True)
            else:
                st.warning("Could not load map data.")
        with tab3:
            self._render_trends_tab(display_data_df.last('24M'), loc_title, self.state.get('selected_industry'))
        with tab4:
            st.subheader("Data Export")
            if display_data_df is not None and not display_data_df.empty:
                st.dataframe(display_data_df.tail(12).style.format("{:.2f}"))
                csv = display_data_df.to_csv().encode('utf-8')
                st.download_button("Download Full Data as CSV", data=csv, file_name=f"labor_pulse_{loc_title}.csv", mime='text/csv')

if __name__ == "__main__":
    app = LaborMarketApp()
    app.run()
