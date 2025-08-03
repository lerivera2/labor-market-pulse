# Labor-Market Pulse: A Streamlit Dashboard
# This script fetches, processes, and visualizes U.S. and State-level labor market data.
# Data Sources: Bureau of Labor Statistics (BLS) API
# Series:
# 1. JTS...JOL: JOLTS Job Openings
# 2. LNS... or LAS... : Unemployment Rate

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

# API Endpoint
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# --- State and Series ID Mappings ---
# Dictionary mapping state names to their FIPS codes for BLS API calls.
STATE_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06',
    'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'Florida': '12', 'Georgia': '13',
    'Hawaii': '15', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19', 'Kansas': '20',
    'Kentucky': '21', 'Louisiana': '22', 'Maine': '23', 'Maryland': '24', 'Massachusetts': '25',
    'Michigan': '26', 'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29', 'Montana': '30',
    'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
    'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40',
    'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46',
    'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '52',
    'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
}

def get_series_ids(location="U.S. Total"):
    """
    Returns the appropriate BLS series IDs based on the selected location.
    """
    if location == "U.S. Total":
        return {
            "Job Openings": "JTS000000000000000JOL",
            "Unemployment Rate": "LNS14000000"
        }
    else:
        fips = STATE_FIPS[location]
        # State-level JOLTS (Job Openings) and LAUS (Unemployment) series IDs
        return {
            "Job Openings": f"JTS{fips}000000000JOL",
            "Unemployment Rate": f"LASST{fips}0000000000003"
        }

# --- Data Fetching and Processing ---

@st.cache_data(ttl=600) # Cache data for 10 minutes (600 seconds)
def get_bls_data(location):
    """
    Fetches data from the BLS API for the last 24 months for a given location.

    Args:
        location (str): The selected location (e.g., "U.S. Total" or a state name).

    Returns:
        pandas.DataFrame: A DataFrame with a 'Date' index and columns for each series.
                         Returns None if the API call fails.
    """
    series_ids = get_series_ids(location)
    
    # Calculate start and end years for the API query (last 24 months)
    end_year = date.today().year
    start_year = (date.today() - timedelta(days=24 * 30)).year

    # Construct the API request payload
    headers = {'Content-type': 'application/json'}
    data = {
        "seriesid": list(series_ids.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": BLS_API_KEY
    }

    try:
        # Make the API request
        response = requests.post(BLS_API_URL, json=data, headers=headers)
        response.raise_for_status()
        json_data = response.json()

        if json_data['status'] != 'REQUEST_SUCCEEDED':
            st.error(f"BLS API Error for {location}: {json_data.get('message', ['Unknown error'])[0]}")
            return None

        # --- Process the data into a DataFrame ---
        all_series_data = []
        for series_name, series_id in series_ids.items():
            # Find the correct series data by matching the seriesID
            for result_series in json_data['Results']['series']:
                if result_series['seriesID'] == series_id:
                    series_data = result_series['data']
                    if not series_data:
                        st.warning(f"No data returned for '{series_name}' in {location}.")
                        continue
                    df = pd.DataFrame(series_data)
                    df['date'] = pd.to_datetime(df['year'] + '-' + df['periodName'])
                    df.set_index('date', inplace=True)
                    df['value'] = pd.to_numeric(df['value'])
                    df.rename(columns={'value': series_name}, inplace=True)
                    all_series_data.append(df[[series_name]])
                    break
        
        if not all_series_data:
            st.error(f"Could not parse any series from the BLS API response for {location}.")
            return None
            
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

def create_dual_axis_plot(df, location):
    """
    Creates a dual-axis Plotly figure for Job Openings and Unemployment Rate
    with an enhanced, professional visual style for a specific location.

    Args:
        df (pandas.DataFrame): DataFrame containing the data to plot.
        location (str): The location name for the chart title.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    # Define a more professional color palette
    colors = {
        'openings': '#1f77b4',  # Muted blue
        'unemployment': '#ff7f0e',  # Safety orange
        'grid': '#e0e0e0',
        'text': '#333333',
        'background': '#f8f8f8'
    }

    # Add Job Openings trace (in thousands) to the primary y-axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Job Openings'] / 1000, # Display in thousands for readability
        name='Job Openings (in thousands)',
        mode='lines+markers',
        line=dict(color=colors['openings'], width=3),
        marker=dict(size=7, symbol='circle'),
        hovertemplate='<b>Job Openings:</b> %{y:,.0f}K<extra></extra>'
    ))

    # Add Unemployment Rate trace to the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Unemployment Rate'],
        name='Unemployment Rate (%)',
        mode='lines+markers',
        line=dict(color=colors['unemployment'], width=2.5, dash='dash'),
        marker=dict(size=8, symbol='x-thin'),
        yaxis='y2',
        hovertemplate='<b>Unemployment Rate:</b> %{y:.1f}%<extra></extra>'
    ))

    # --- Update layout and axes for a modern, clean look ---
    title_text = f'<b>Labor Market Pulse for {location}</b>'
    fig.update_layout(
        title=dict(
            text=title_text,
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color=colors['text'])
        ),
        xaxis=dict(
            title_text='Date',
            showgrid=True, 
            gridcolor=colors['grid'],
            gridwidth=1,
            linecolor=colors['grid'],
            tickfont=dict(size=12, color=colors['text'])
        ),
        yaxis=dict(
            title=dict(text='<b>Job Openings (in thousands)</b>', font=dict(color=colors['openings'], size=14)),
            tickfont=dict(color=colors['openings'], size=12),
            showgrid=True,
            gridcolor=colors['grid'],
            gridwidth=1
        ),
        yaxis2=dict(
            title=dict(text='<b>Unemployment Rate (%)</b>', font=dict(color=colors['unemployment'], size=14)),
            tickfont=dict(color=colors['unemployment'], size=12),
            anchor='x',
            overlaying='y',
            side='right',
            showgrid=False # Avoid visual clutter from a second grid
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        template='simple_white', # A clean base template
        plot_bgcolor=colors['background'], # Light background for the plot area
        paper_bgcolor='white', # Background for the whole figure
        height=550,
        margin=dict(l=80, r=80, t=100, b=80),
        hovermode='x unified'
    )
    
    # Add a vertical line to highlight the most recent data point
    if not df.empty:
        fig.add_vline(x=df.index[-1], line_width=2, line_dash="dot", line_color="grey")

        # Add a separate annotation for the line to avoid the TypeError
        fig.add_annotation(
            x=df.index[-1],
            y=1.05, # Position it slightly above the plot area
            yref="paper", # Use 'paper' coordinates for y to place it relative to the plot area
            text="Latest Data",
            showarrow=False,
            xanchor="right",
            font=dict(
                size=12,
                color="grey"
            )
        )

    return fig

# --- Streamlit App Layout ---

st.set_page_config(page_title="Labor-Market Pulse", layout="wide")

st.title("Labor-Market Pulse Dashboard")

# --- Sidebar for user input ---
st.sidebar.header("Dashboard Controls")
location_list = ["U.S. Total"] + sorted(STATE_FIPS.keys())
selected_location = st.sidebar.selectbox(
    "Select a Location:",
    location_list
)

st.sidebar.info(
    """
    **About the Data:**
    - **Job Openings (JOLTS):** Measures unmet labor demand.
    - **Unemployment Rate:** Measures the share of the workforce that is jobless.
    - **Source:** [U.S. Bureau of Labor Statistics (BLS)](https://www.bls.gov/data/)
    """
)

# --- Main Panel ---
st.markdown(f"Displaying data for **{selected_location}**.")

# Button to manually refresh the data
if st.button('Refresh Data'):
    # Caching is time-based, but this allows a manual override/update.
    st.cache_data.clear()

# Fetch and display data for the selected location
data_df = get_bls_data(selected_location)

if data_df is not None and not data_df.empty:
    # Display the plot
    st.plotly_chart(create_dual_axis_plot(data_df, selected_location), use_container_width=True)

    # Display latest data points
    st.subheader("Latest Data Points")
    latest_openings = data_df['Job Openings'].iloc[-1]
    latest_unemployment = data_df['Unemployment Rate'].iloc[-1]
    latest_date = data_df.index[-1].strftime('%B %Y')

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Total Job Openings ({latest_date})",
            value=f"{latest_openings/1_000_000:.2f}M" if latest_openings >= 1000000 else f"{latest_openings/1000:,.0f}K",
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
    st.warning(f"Could not retrieve or process data for {selected_location}. This can happen if data for a specific series is not available for the selected period.")
