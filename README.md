# Labor Market Pulse

An interactive dashboard for analyzing U.S. labor market data from the Bureau of Labor Statistics (BLS), built with Streamlit. This project serves as a practical application of Python for data analysis and visualization as part of a learning journey.

## Features

* **Key Metrics:** Instantly view the Unemployment Rate, Job Openings, and Quits Rate with month-over-month trend indicators.
* **Smart Filtering:** Analyze data for the `U.S. Total` or individual `States`. At the national level, you can drill down by `Industry`.
* **Interactive Charts:** Explore historical trends with a date slider, a national choropleth map, and dynamic time-series charts.
* **Data Export:** Download any filtered dataset as a CSV.

## Tech Stack

* **Framework:** Streamlit
* **Data:** pandas, requests
* **Visualization:** Plotly
* **Source:** U.S. Bureau of Labor Statistics (BLS) API v2

## Getting Started

### Prerequisites

* A free API key from the BLS Public Data API page.

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/labor-market-pulse.git](https://github.com/YOUR_USERNAME/labor-market-pulse.git)
    cd labor-market-pulse
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add your API Key:**
    Create a file named `.streamlit/secrets.toml` and add your key:
    ```toml
    BLS_API_KEY = "YOUR_API_KEY_HERE"
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

### Deploy to Streamlit Cloud

1.  Push your project to a public GitHub repository.
2.  Sign in to share.streamlit.io and create a "New app" from your repository.
3.  In the "Advanced settings," add your `BLS_API_KEY` as a secret.
4.  Click "Deploy!"

## License

This project is licensed under the MIT License.
