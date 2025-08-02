# Labor-Market Pulse: A Streamlit Dashboard
# This script fetches, processes, and visualizes U.S. labor market data.
# Data Sources: Bureau of Labor Statistics (BLS) API
# Series:
# 1. JTS000000000000000JOL: JOLTS Job Openings, Total Nonfarm
# 2. LNS14000000: Unemployment Rate, Civilian Labor Force

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import os

# --- Configuration ---
# It's recommended to store your API key as a Streamlit secret.
# For local development, you can set it as an environment variable.
BLS_API_KEY = os.environ.get("BLS_API_KEY")
if not BLS_API_KEY:
    # Fallback for when the secret isn't set in Streamlit Cloud, allowing for local testing.
    try:
        BLS_API_KEY = st.secrets["BLS_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("BLS_API_KEY not found. Please set it as a Streamlit secret or environment variable.")
        st.stop()


# Series IDs for the BLS API
SERIES_IDS = {
    "Job Openings": "JTS000000000000000JOL",
    "Unemployment Rate": "LNS14000000"
}

# API Endpoint
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# --- Data Fetching and Processing ---

@st.cache_data(ttl=600) # Cache data for 10 minutes (600 seconds)
def get_bls_data(series_ids, api_key):
    """
    Fetches data from the BLS API for the last 24 months.

    Args:
        series_ids (dict): A dictionary mapping descriptive names to BLS series IDs.
        api_key (str): Your BLS API key.

    Returns:
        pandas.DataFrame: A DataFrame with a 'Date' index and columns for each series.
                         Returns None if the API call fails.
    """
    # Calculate start and end years for the API query (last 24 months)
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=24 * 30)).year

    # Construct the API request payload
    headers = {'Content-type': 'application/json'}
    data = {
        "seriesid": list(series_ids.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": api_key
    }

    try:
        # Make the API request
        response = requests.post(BLS_API_URL, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        json_data = response.json()

        if json_data['status'] != 'REQUEST_SUCCEEDED':
            st.error(f"BLS API Error: {json_data.get('message', ['Unknown error'])[0]}")
            return None

        # --- Process the data into a DataFrame ---
        all_series_data = []
        for series_name, series_id in series_ids.items():
            series_data = json_data['Results']['series'][list(series_ids.values()).index(series_id)]['data']
            df = pd.DataFrame(series_data)
            df['date'] = pd.to_datetime(df['year'] + '-' + df['periodName'])
            df.set_index('date', inplace=True)
            df['value'] = pd.to_numeric(df['value'])
            df.rename(columns={'value': series_name}, inplace=True)
            all_series_data.append(df[[series_name]])

        # Combine the series into a single DataFrame
        combined_df = pd.concat(all_series_data, axis=1)
        combined_df.sort_index(ascending=True, inplace=True)

        # The API might return more than 24 months, so we trim it here
        last_24_months_df = combined_df.last('24M')
        return last_24_months_df.ffill() # Forward fill to handle any missing data points

    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data from BLS API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- Plotting Function ---

def create_dual_axis_plot(df):
    """
    Creates a dual-axis Plotly figure for Job Openings and Unemployment Rate.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to plot.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    # Add Job Openings trace (in thousands) to the primary y-axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Job Openings'] / 1000, # Display in thousands for readability
        name='Job Openings (in thousands)',
        mode='lines+markers',
        line=dict(color='royalblue', width=3),
        marker=dict(size=6)
    ))

    # Add Unemployment Rate trace to the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Unemployment Rate'],
        name='Unemployment Rate (%)',
        mode='lines+markers',
        line=dict(color='firebrick', width=3, dash='dash'),
        marker=dict(symbol='cross', size=8),
        yaxis='y2'
    ))

    # --- Update layout and axes ---
    fig.update_layout(
        title_text='<b>Labor Market Pulse: Job Openings vs. Unemployment Rate</b>',
        title_x=0.5,
        xaxis_title='Date',
        yaxis=dict(
            title='<b>Job Openings (in thousands)</b>',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue')
        ),
        yaxis2=dict(
            title='<b>Unemployment Rate (%)</b>',
            titlefont=dict(color='firebrick'),
            tickfont=dict(color='firebrick'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=80, r=80, t=80, b=50)
    )
    return fig

# --- Streamlit App Layout ---

st.set_page_config(page_title="Labor-Market Pulse", layout="wide")

st.title("Labor-Market Pulse Dashboard")
st.markdown("""
This dashboard provides a near real-time view of key U.S. labor market indicators.
Data is sourced from the Bureau of Labor Statistics (BLS) and updated every 10 minutes.
""")

# Button to manually refresh the data
if st.button('Refresh Data'):
    # Caching is time-based, but this allows a manual override/update.
    # To force a refresh, we can clear the cache for the specific function.
    st.cache_data.clear()

# Fetch and display data
data_df = get_bls_data(SERIES_IDS, BLS_API_KEY)

if data_df is not None and not data_df.empty:
    # Display the plot
    st.plotly_chart(create_dual_axis_plot(data_df), use_container_width=True)

    # Display latest data points
    st.subheader("Latest Data Points")
    latest_openings = data_df['Job Openings'].iloc[-1]
    latest_unemployment = data_df['Unemployment Rate'].iloc[-1]
    latest_date = data_df.index[-1].strftime('%B %Y')

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Total Job Openings ({latest_date})",
            value=f"{latest_openings/1_000_000:.2f}M",
            help="Total nonfarm job openings at the end of the month, seasonally adjusted."
        )
    with col2:
        st.metric(
            label=f"Unemployment Rate ({latest_date})",
            value=f"{latest_unemployment}%",
            help="The percentage of the total labor force that is unemployed but actively seeking employment."
        )

    # Display the raw data in an expandable section
    with st.expander("View Raw Data Table"):
        # Format the DataFrame for better display
        display_df = data_df.copy()
        display_df['Job Openings'] = display_df['Job Openings'].apply(lambda x: f"{x:,.0f}")
        display_df['Unemployment Rate'] = display_df['Unemployment Rate'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Could not retrieve or process data. Please check the API key and network connection.")

st.sidebar.info(
    """
    **About the Data:**
    - **Job Openings (JOLTS):** Measures unmet labor demand. A high number suggests a tight labor market where employers struggle to find workers.
    - **Unemployment Rate:** Measures the share of the workforce that is jobless. A low number indicates a strong labor market.
    - **Source:** [U.S. Bureau of Labor Statistics (BLS)](https://www.bls.gov/data/)
    """
)

