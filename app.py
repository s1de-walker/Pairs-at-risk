# Libraries
# ------------------------------------
import streamlit as st
import pandas as pd
import re
from datetime import date, timedelta

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Import the statsmodels module for regression and the adfuller function
# Import statsmodels.formula.api
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Factor & Sector Lab", page_icon="📊", layout="wide")

from datetime import date, timedelta
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
      h1 {letter-spacing: -0.02em;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stTabs"] button {font-size: 1rem; padding: 0.5rem 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_FACTORS = [
    "EQUAL50ADD", "MOVALUE", "LOWVOLIETF", "HDFCQUAL",
    "MOMOMENTUM", "HDFCGROWTH", "DIVOPPBEES", "SELECTIPO",
]
DEFAULT_SECTORS = [
    "CHEMICAL", "CPSEETF", "GROWWRAIL", "MAKEINDIA", "MNC", "COMMOIETF",
    "INFRA", "DEFENCE", "INTERNET", "EVIETF", "CONSUMBEES", "NIFTYBEES",
    "GOLDBEES", "BANKBEES", "HEALTHIETF", "GROWWPOWER", "ENERGY", "OILIETF",
    "GROWWRLTY", "METAL", "MASPTOP50", "MON100", "MONQ50", "HNGSNGBEES",
]
HORIZONS = {"Weekly (5d)": 5, "15 days": 15, "Monthly (21d)": 21}
OPS = {
    ">": np.greater, ">=": np.greater_equal,
    "<": np.less, "<=": np.less_equal,
}


def parse_tickers(text: str) -> list[str]:
    items = [t.strip().upper() for t in re.split(r"[,\n\s]+", text) if t.strip()]
    return list(dict.fromkeys(items))  # de-dupe, keep order

import altair as alt

# Set app to wide mode
st.set_page_config(layout="wide")

st.title("Pairs Watch")
st.caption("Monitor your pairs with risk metrics and alerts.")
st.write("")

col10, colmid, col20 = st.columns((1,0.1,1))
with col10:
    
    # Input
    # ------------------------------------
    # Title
    
    
    if 'pairs' not in st.session_state:
        st.session_state.pairs = []
    
    st.subheader("Enter the date range for Technical analysis")
    
    # Date Input Section
    col_date1, col_date2 = st.columns(2)
    
    # Default values (1-year difference)
    default_start = (datetime.today() - timedelta(days=760)).strftime('%Y-%m-%d')
    default_end = datetime.now().strftime('%Y-%m-%d')
    
    # Take user inputs for start and end date
    
    start_date = col_date1.date_input("Start Date", datetime.strptime(default_start, '%Y-%m-%d'))
    end_date = col_date2.date_input("End Date", datetime.strptime(default_end, '%Y-%m-%d'))
    
    # Ensure start_date is before end_date
    if start_date >= end_date:
        st.error("❗ Start Date must be before End Date!")
        st.stop()  # Stops execution immediately after showing error
    
    # Calculate month difference
    st.caption(f"Selected period: **{(end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)} months**")
    
    date_range_days = (end_date - start_date).days  # Calculate total available days
    
    st.write("")
    st.subheader("Enter the allocation and tickers")
    st.caption("as per yfinance convention")
    
    # Form for user input
    
    with st.form("pairs_form"):
        col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 3])
        
        units1 = col1.number_input("Lots", min_value=1, step=1, key="units1")
        ticker1 = col2.text_input("Stock/ETF 1", value="JPM").upper()
        with col3:
            st.write("")
            st.markdown("<p style='text-align: center; font-size: 24px;'>_</p>", unsafe_allow_html=True)
        units2 = col4.number_input("Lots", min_value=1, step=1, key="units2")
        ticker2 = col5.text_input("Stock/ETF 2",  value="BAC").upper()
        
        submit = st.form_submit_button("Confirm Pair")
    
        if submit and ticker1 == ticker2:
                st.error("❗ Error: Both tickers cannot be the same! Please select different stocks or ETFs.")
                st.stop()  # Stops execution immediately after showing error
    
        
        st.session_state.pairs.append({"Units 1": units1, "Stock/ETF 1": ticker1, "Units 2": units2, "Stock/ETF 2": ticker2})
    
        
    st.divider()
    # Input End------------------------------------
    
    # Signal initiation
    # =================
    
    price_ratio_signal = 0
    st_vol_signal = 0
    range_signal = 0
    coint_signal = 0
    
    # Fetching data
    # ------------------------------------
    # Historical Time Series Calculation
    if st.session_state.pairs:
    
        try:
            # Fetch historical data and create required data
            data = yf.download([ticker1, ticker2], start=start_date, end=end_date+ timedelta(days=1))["Close"]
            data = data[[ticker1,ticker2]]
            data['Price Ratio'] = data[ticker1]/data[ticker2]
            data["Pair Value"] = data[ticker1]*units1 - data[ticker2]*units2
            returns = data.pct_change().dropna()
            #cm_returns = (returns + 1).cumprod() - 1
    
            data_high = yf.download([ticker1, ticker2], start=start_date, end=end_date+ timedelta(days=1))["High"] 
            data_low = yf.download([ticker1, ticker2], start=start_date, end=end_date+ timedelta(days=1))["Low"]
            
            # Check if data is empty (invalid ticker)
            if data.empty or ticker1 not in data.columns or ticker2 not in data.columns:
                st.error("❗ Error: One or both tickers are invalid. Please enter correct stock/ETF symbols.")
                st.stop()  # Stop execution if tickers are invalid
    
            # Check if the DataFrame is not empty and the index is within range
            if data.empty or returns.empty:
                st.error("DataFrames are empty. Please check your data source.")
                st.stop()
            
            if len(data[ticker1]) == 0 or len(returns[ticker1]) == 0:
                st.error(f"No data available for {ticker1}. Please check your data source.")
                st.stop()
            
        except Exception as e:
            st.error(f"❗ Error fetching historical data: {e}")
            st.stop()  # Stops execution immediately after showing error
            
        #returns = data[[ticker1, ticker2]].pct_change().dropna()
        
    
        # Market Summary
        # ------------------------------------
        with st.expander(f"Market Summary"):
            col1, col2 = st.columns(2)
            col1.metric(f"{ticker1}", f"${data[ticker1].iloc[-1]:.2f}", f"{returns[ticker1].iloc[-1] * 100:.2f}%")
            col2.metric(f"{ticker2}", f"${data[ticker2].iloc[-1]:.2f}", f"{returns[ticker2].iloc[-1] * 100:.2f}%")
            
            returns_cr = data[[ticker1,ticker2]].pct_change().dropna()
            cm_returns = (returns_cr + 1).cumprod() - 1
            
            # Define custom colors
            color_map = {
                cm_returns.columns[0]: "#fb580d",  # Fiery Orange
                cm_returns.columns[1]: "#5cc8e2",  # Electric Blue
            }
    
            #st.dataframe(cm_returns)
    
            fig = px.line(
            cm_returns.reset_index(),  # Ensure Date is a column
            x="Date",  # ✅ Use "Date" as the x-axis
            y=cm_returns.columns,  # ✅ Use all columns except "Date" for y-axis
            title="Cumulative Returns",
            color_discrete_map=color_map
            )
        
            st.plotly_chart(fig)
            
            
            #st.dataframe(data)
            #st.dataframe(returns)
        # Price Ratio
        # ------------------------------------
        with st.expander(f"Price Ratio"):
            data_pr = data.dropna()
            mean_ratio = data_pr['Price Ratio'].mean()
            col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 3])
            percentile = col1.number_input("Select Percentile:", min_value=50.00, max_value=99.99, value=95.00, format="%.2f")
            lower_bound = np.percentile(data_pr['Price Ratio'], 100-percentile)
            upper_bound = np.percentile(data_pr['Price Ratio'], percentile)
        
            fig = px.line(data_pr, x=data_pr.index, y='Price Ratio', title=f"Price Ratio ({ticker1}/{ticker2})", line_shape='linear')
            fig.update_traces(line=dict(color='#A55B4B'))  # Custom orange-red color for better contrast
            
            # Mean line with annotation on the left
            fig.add_hline(y=mean_ratio, line_dash="dot", line_color="white", line_width=1.5)
            fig.add_annotation(
                x=data_pr.index.min(), y=mean_ratio, text="Mean",
                showarrow=False, xanchor="left", font=dict(color="grey", size=10), bgcolor="black"
            )
        
            # Lower Bound with annotation on the left
            fig.add_hline(y=lower_bound, line_dash="solid", line_color="#F2F2F2", line_width=1.5)
            fig.add_annotation(
                x=data_pr.index.min(), y=lower_bound, text=f"{percentile}th Percentile",
                showarrow=False, xanchor="left", font=dict(color="grey", size=10), bgcolor="black"
            )
        
            # Upper Bound with annotation on the left
            fig.add_hline(y=upper_bound, line_dash="solid", line_color="#F2F2F2", line_width=1.5)
            fig.add_annotation(
                x=data_pr.index.min(), y=upper_bound, text=f"{100 - percentile}th Percentile",
                showarrow=False, xanchor="left", font=dict(color="grey", size=10), bgcolor="black"
            )
            # Add a translucent film from lower bound to upper bound with custom color
            fig.add_shape(
                type="rect",
                x0=data_pr.index.min(),
                x1=data_pr.index.max(),
                y0=lower_bound,
                y1=upper_bound,
                fillcolor="rgba(64,64,64, 0.3)",  # Custom color with 30% opacity
                line=dict(width=0)
            )
        
            st.plotly_chart(fig)
    
            st.dataframe(data)
            
            if data_pr['Price Ratio'].iloc[-1] < lower_bound:
                st.success("➕ Long Signal: Price Ratio below lower bound")
                price_ratio_signal = 1
            elif data_pr['Price Ratio'].iloc[-1] > upper_bound:
                st.warning("➖ Short Signal: Price Ratio above upper bound")
                price_ratio_signal = -1
                
        # Pair spread
        # ------------------------------------
        with st.expander(f"Pair Spread"):
            fig_spread = px.line(data_pr, x=data_pr.index, y='Pair Value', title=f"Pair Spread: {units1} {ticker1} - {units2} {ticker2}", line_shape='linear')
            fig_spread.update_traces(line=dict(color='#4A4A4A'))  
            st.plotly_chart(fig_spread)
    
        # Pair spread
        # ------------------------------------
        with st.expander(f"Rolling Volatility Ratio"):
            st.subheader("Is it getting riskier?")
            st.caption("Select rolling windows for short-term and long-term volatility.")
            
            col1, col2 = st.columns(2)
            # Proceed with your number input
            with col1:
                short_vol_window = st.number_input("Short-Term Window (Days):", min_value=1, max_value=date_range_days, value=10)
            with col2:
                long_vol_window = st.number_input("Long-Term Window (Days):", min_value=1, max_value=date_range_days, value=50)
    
            # User input for percentile value
            col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 3])
            percentile2 = col1.number_input("Select Percentile:", min_value=50.00, max_value=99.99, value=90.00, format="%.2f", key="percentile_input")
                
            
            # Calculate rolling volatility for each stock
            rolling_volatility_ticker1_short = returns[ticker1].rolling(window=short_vol_window).std().dropna()*units1
            rolling_volatility_ticker2_short = returns[ticker2].rolling(window=short_vol_window).std().dropna()*units2
    
            # Calculate long-term rolling volatility for each stock
            rolling_volatility_ticker1_long = returns[ticker1].rolling(window=long_vol_window).std().dropna() * units1
            rolling_volatility_ticker2_long = returns[ticker2].rolling(window=long_vol_window).std().dropna() * units2
    
            #st.dataframe(rolling_volatility_ticker1)
            #st.dataframe(rolling_volatility_ticker2)
    
            # Calculate rolling volatility ratio (ticker1 / ticker2)
            #rolling_volatility_ratio = rolling_volatility_ticker1 / rolling_volatility_ticker2
            
            # Rename column for the ratio
            #rolling_volatility_ratio = rolling_volatility_ratio.rename('Rolling Volatility Ratio')
            #st.dataframe(rolling_volatility_ratio)
    
            # Calculate rolling volatility ratio (ticker1 / ticker2)
            rolling_volatility_ratio_short = rolling_volatility_ticker1_short / rolling_volatility_ticker2_short
            rolling_volatility_ratio_long = rolling_volatility_ticker1_long / rolling_volatility_ticker2_long
            
            # Drop NaN values to start the chart from where the data is available
            rolling_volatility_ratio_short = rolling_volatility_ratio_short.dropna()
            rolling_volatility_ratio_long = rolling_volatility_ratio_long.dropna()
            
            
    
            # Create a DataFrame for plotting
            rolling_volatility_df = pd.DataFrame({
                'Date': rolling_volatility_ratio_short.index,
                'Rolling Volatility Ratio (Short-Term)': rolling_volatility_ratio_short.values,
                'Rolling Volatility Ratio (Long-Term)': rolling_volatility_ratio_long.reindex(rolling_volatility_ratio_short.index).values
            })
        
            # Create Plotly figure for rolling volatility ratio
            fig_volatility_ratio = px.line(
                rolling_volatility_df,
                x='Date',
                y=['Rolling Volatility Ratio (Short-Term)', 'Rolling Volatility Ratio (Long-Term)'],
                title=f"Rolling Volatility Ratio ({units1}.{ticker1} / {units2}.{ticker2})",
                labels={'value': 'Volatility Ratio', 'variable': 'Rolling Volatility Type'},
                color_discrete_map={
                    'Rolling Volatility Ratio (Short-Term)': 'red',
                    'Rolling Volatility Ratio (Long-Term)': 'grey'
                }
            )
            
            # Update layout for legend position and other customizations
            fig_volatility_ratio.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    title_text=None  # This removes the legend title
                )
            )
    
            
            # Show chart in Streamlit
            st.plotly_chart(fig_volatility_ratio)
    
    
            # ROLLING VOLATILITY GAP
            # Ensure both series have the same indexes
            rolling_volatility_ratio_long_aligned = rolling_volatility_ratio_long.reindex(rolling_volatility_ratio_short.index)
            
            # Calculate the gap between short-term and long-term rolling volatility ratios
            volatility_ratio_gap = rolling_volatility_ratio_short - rolling_volatility_ratio_long_aligned
            
            # Create a DataFrame for plotting the gap
            volatility_ratio_gap_df = pd.DataFrame({
                'Date': volatility_ratio_gap.index,
                'Volatility Ratio Gap': volatility_ratio_gap.values
            })
    
            volatility_ratio_gap_df = volatility_ratio_gap_df.dropna()
            
            lower_bound2 = np.percentile(volatility_ratio_gap_df['Volatility Ratio Gap'], 100-percentile2)
            upper_bound2 = np.percentile(volatility_ratio_gap_df['Volatility Ratio Gap'], percentile2)
            
            
            # Create Plotly figure for volatility ratio gap
            fig_volatility_ratio_gap = px.line(
                volatility_ratio_gap_df,
                x='Date',
                y='Volatility Ratio Gap',
                title=f"Gap Between Short-Term and Long-Term Volatility Ratios ({units1}.{ticker1} / {units2}.{ticker2})",
                labels={'Volatility Ratio Gap': 'Volatility Ratio Gap'}
            )
            fig_volatility_ratio_gap.update_traces(line=dict(color='#A55B4B'))  
    
            # Add horizontal lines for percentiles and mean
            fig_volatility_ratio_gap.add_hline(y=upper_bound2, line_dash="solid", line_color="grey")
            fig_volatility_ratio_gap.add_hline(y=lower_bound2, line_dash="solid", line_color="grey")
            mean_value = volatility_ratio_gap_df['Volatility Ratio Gap'].mean()
            fig_volatility_ratio_gap.add_hline(y=mean_value, line_dash="dot", line_color="grey")
                
        
            # Update layout for legend position and other customizations
            fig_volatility_ratio_gap.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    title_text=None  # This removes the legend title
                )
            )
    
            # Add a translucent film from lower bound to upper bound with custom color
            fig_volatility_ratio_gap.add_shape(
                type="rect",
                x0=volatility_ratio_gap_df["Date"].min(),
                x1=volatility_ratio_gap_df["Date"].max(),
                y0=lower_bound2,
                y1=upper_bound2,
                fillcolor="rgba(64,64,64, 0.7)",  # Custom color with 30% opacity
                line=dict(width=0)
            )
    
            # Show chart in Streamlit
            st.plotly_chart(fig_volatility_ratio_gap)
    
    
            # Check if the spread crosses either of the thresholds
            if volatility_ratio_gap_df['Volatility Ratio Gap'].iloc[-1] > upper_bound2 or volatility_ratio_gap_df['Volatility Ratio Gap'].iloc[-1] < lower_bound2:
                st.error("🚨 Warning: Short term volatility has spiked")
                st_vol_signal = 1
    
    
        # Rolling Beta
        # ------------------------------------
        with st.expander(f"Rolling Beta"):
            # Calculate rolling beta
            #st.dataframe(returns)
            returns_rb = returns[[ticker1, ticker2]]
            #st.dataframe(returns_rb)
    
            col1, col2 = st.columns(2)
            # Proceed with your number input
            with col1:
                window = st.number_input("Rolling Window (Days):", min_value=1, max_value=date_range_days, value=10)
    
            # Calculate rolling covariance and variance
            rolling_cov = returns_rb[ticker1].rolling(window).cov(returns_rb[ticker2])
            rolling_var = returns_rb[ticker2].rolling(window).var()
    
            # Calculate rolling beta
            rolling_beta = rolling_cov / rolling_var
            rolling_beta = rolling_beta.dropna()
    
            #st.dataframe(rolling_beta)
    
            fig_rb = px.line(rolling_beta, title=f"Rolling Beta of {ticker1} on {ticker2} ({window}-day window)", color_discrete_sequence=['#A55B4B'])
            fig_rb.update_layout(showlegend=False)  # Remove the legend
    
            # Calculate the mean of the rolling beta
            mean_rolling_beta = rolling_beta.mean()
            
            # Add horizontal line at the mean of the rolling beta
            fig_rb.add_shape(
                type="line",
                x0=rolling_beta.index.min(),
                x1=rolling_beta.index.max(),
                y0=mean_rolling_beta,
                y1=mean_rolling_beta,
                line=dict(color="white", width=2, dash="solid")
            )
            # Add label for the mean line
            fig_rb.add_annotation(
                x=rolling_beta.index.min(),
                y=mean_rolling_beta,
                text="Mean",
                showarrow=False,
                yshift=10,
                font=dict(color="grey")
            )
    
            st.plotly_chart(fig_rb)
    
        # Cointegration
        # ------------------------------------
        with st.expander(f"Cointegration"):
            st.subheader("Cointegration")
            st.caption("Cointegration is a a statistical property. It occurs when 2 non-stationary time series move together in a way that they form a **stationary linear combination** in the long run.")
       
            st.caption("Linear combination:")
            st.write(f"{ticker2} returns = (OLS β) . {ticker1} returns + c + ε")
            st.caption("c = intercept, ε = residuals")
            
            # OLS Regression
            # Prepare independent (X) and dependent (Y) variables
            
            X = returns[ticker1]  # Predictor (Independent variable)
            Y = returns[ticker2]  # Response (Dependent variable)
            
            # Y = β . X + c + error
    
            # Add constant term for intercept
            X = sm.add_constant(X) # This adds a column of ones to the predictor variables X
            
            # Run OLS regression
            model = sm.OLS(Y, X).fit()
    
            # Extract key regression metrics
            r_squared = model.rsquared
            beta = model.params[ticker1]
    
            # Compute ADF test on residual (spread)
            spread = returns[ticker2] - beta * returns[ticker1]
            adf_pvalue = adfuller(spread)[1]
    
            # Create a scatter plot
            fig = px.scatter(returns, x=ticker1, y=ticker2, title=f'Scatter Plot of {ticker1} vs {ticker2} Returns')
            
            # Display the plot in Streamlit
            st.plotly_chart(fig)
    
            # Display results
            # Create three columns
            col1, col2, col3 = st.columns(3)
    
            # Display R-squared in the first column
            col1.metric(label="R-Squared", value=f"{r_squared:.3f}")
            # Use Streamlit's built-in color formatting
            col1.caption(f"*:grey[Relationship Strength: {ticker1} explains {r_squared*100:.0f}% of the variation in {ticker2}]*")
            
            # Display OLS Beta in the second column
            col2.metric(label="OLS Beta", value=f"{beta:.3f}")
            col2.caption(f"*:grey[Effect of {ticker1} on {ticker2}: A 1-unit increase in {ticker1} is associated with a {beta:.2f} increase in {ticker2}.\n]*")
            
            # Display ADF Test P-Value in the third column
            col3.metric(label="ADF P-Value", value=f"{adf_pvalue:.3f}")
            
            if adf_pvalue < 0.05:
                col3.write(f"*✅ The spread is **STATIONARY**.*")
                col3.caption("*:grey[It means the spread has a constant mean and variance over time, suggesting a stable relationship that is likely to revert to its average.]*")
            else:
                col3.caption(f"*❌ The spread is **non-stationary** (p-value: {adf_pvalue:.3f})*")
    
    
    
            st.write(" ")
            st.subheader("Cointegration Residuals")
            st.markdown("""
            ###### <span style="color:#A55B4B">Cointegration residuals</span> = Deviations of actual values from predicted relationship
            """, unsafe_allow_html=True)
            #-------------------------------------------------------------------------
            st.caption("*How far the spread is from its **\"fair value\"** as per the stationary linear relationship.*")
            
            # Compute the cointegration residuals
            #-------------------------------------------------------------------------
            df_coint = model.resid
            
            # Convert residuals to DataFrame for plotting
            df_coint_plot = pd.DataFrame({"Time": returns.index, "Residuals": df_coint})
    
            # User input for percentile value
            col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 3])
            percentile3 = col1.number_input("Select Percentile:", min_value=50.00, max_value=99.99, value=95.00, format="%.2f", key="percentile_input_cr")
    
            lower_bound3 = np.percentile(df_coint_plot['Residuals'], 100-percentile3)
            upper_bound3 = np.percentile(df_coint_plot['Residuals'], percentile3)
    
            # Plot the residuals with custom color
            fig_cr = px.line(df_coint_plot, x="Time", y="Residuals", title="Cointegration Residuals", color_discrete_sequence=['#A55B4B'])
    
            # Add a grey horizontal dotted line at y=0
            fig_cr.add_shape(
                type="line",
                x0=df_coint_plot["Time"].min(),
                x1=df_coint_plot["Time"].max(),
                y0=0,
                y1=0,
                line=dict(color="white", width=2, dash="dot")
            )
    
    
            # Add horizontal lines for percentiles and mean
            fig_cr.add_hline(y=upper_bound3, line_dash="solid", line_color="white")
            fig_cr.add_hline(y=lower_bound3, line_dash="solid", line_color="white")
    
            # Add a translucent film from lower bound to upper bound with custom color
            fig_cr.add_shape(
                type="rect",
                x0=df_coint_plot["Time"].min(),
                x1=df_coint_plot["Time"].max(),
                y0=lower_bound3,
                y1=upper_bound3,
                fillcolor="rgba(64,64,64, 0.7)",  # Custom color with 30% opacity
                line=dict(width=0)
            )
            
            # Display the plot in Streamlit
            st.plotly_chart(fig_cr)
    
    
            # Trading signal
            if df_coint.iloc[-1] < lower_bound3:
                st.error("➖ Trading Signal: Sell (Residuals are below the lower threshold)")
                coint_signal = 1
                
            elif df_coint.iloc[-1] > upper_bound3:
                st.success("➕ Trading Signal: Buy (Residuals are above the upper threshold)")
                coint_signal = -1
    
    
        # Range Ratio
        # ------------------------------------
        with st.expander(f"Range Difference"):
            #st.write("Range Ratio")
            #st.dataframe(data_high)
            #st.dataframe(data_low)
            st.write("Range = High - Low")
            range1 = data_high[ticker1] - data_low[ticker1]
            range2 = data_high[ticker2] - data_low[ticker2]
            #valid_days = (range1>0) & (range2>0)
            #st.dataframe(valid_days)
            
            range_diff = range1-range2
            #range_ratio_filtered = range_ratio[valid_days]
    
            # Convert ratios to DataFrame for plotting
            df_range_diff = pd.DataFrame({"Time": range_diff.index, "Range Difference": range_diff})
            df_range_diff = df_range_diff.dropna()
    
            #st.dataframe(range1)
            #st.dataframe(range2)
            #st.dataframe(range_ratio_filtered)
            # Plot the residuals with custom color
            fig_rr = px.line(df_range_diff, x="Time", y="Range Difference", title=f"Range of {ticker1} - Range of {ticker2}", color_discrete_sequence=['#A55B4B'])
            
            
            # User input for percentile value
            col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 3])
            percentile4 = col1.number_input("Select Percentile:", min_value=50.00, max_value=99.99, value=95.00, format="%.2f", key="percentile_input_rrr")
    
            lower_bound4 = np.percentile(df_range_diff['Range Difference'], 100-percentile4)
            upper_bound4 = np.percentile(df_range_diff['Range Difference'], percentile4)
    
            # Add horizontal lines for percentiles and mean
            fig_rr.add_hline(y=upper_bound4, line_dash="solid", line_color="white")
            fig_rr.add_hline(y=lower_bound4, line_dash="solid", line_color="white")
    
            # Add a translucent film from lower bound to upper bound with custom color
            fig_rr.add_shape(
                type="rect",
                x0=df_range_diff["Time"].min(),
                x1=df_range_diff["Time"].max(),
                y0=lower_bound4,
                y1=upper_bound4,
                fillcolor="rgba(64,64,64, 0.7)",  # Custom color with 30% opacity
                line=dict(width=0)
            )        
    
            
            # Display the plot in Streamlit
            st.plotly_chart(fig_rr)
            #st.dataframe(df_range_diff)
    
            # Trading signal
            if range_diff.iloc[-1] < lower_bound4:
                st.error(f"🚨 Range of {ticker2} spiking")
                range_signal = 1
            elif range_diff.iloc[-1] > upper_bound4:
                st.error(f"🚨 Range of {ticker1} spiking")
                range_signal = -1
    
    
    
    # Signals
    # =======
    
    if price_ratio_signal == 1:
        st.success("➕ Long Signal: Price Ratio below lower bound")
        st.write("Check **Price Ratio**")
    elif price_ratio_signal == -1:
        st.warning("➖ Short Signal: Price Ratio above upper bound")
        st.write("Check **Price Ratio**")
    
    if st_vol_signal == 1:
        st.error("🚨 Warning: Short term volatility has spiked")
        st.write("Check **Rolling Volatility Ratio**")
    
    if range_signal == 1:
        st.error(f"🚨 Range of {ticker2} spiking")
        st.write("Check **Range Difference**")
    elif range_signal == -1:
        st.error(f"🚨 Range of {ticker1} spiking")
        st.write("Check **Range Difference**")
    
    if coint_signal == 1:
        st.warning(f"➖ Trading Signal: Short pair: {ticker2} undervalued or {ticker1} overvalued")
        st.write("Check **Cointegration**")
    elif coint_signal == -1:
        st.success(f"➕ Trading Signal: Long pair: {ticker1} undervalued or {ticker2} overvalued")
        st.write("Check **Cointegration**")
        
with colmid:
    st.write("")

with col20:
    st.header("Season Check")
    st.divider()

    if 'pairs_seasonality' not in st.session_state:
            st.session_state.pairs_seasonality = []
        
    st.subheader("Enter the date range for Seasonality analysis")
    col_sd, col_ed = st.columns(2)

    with col_sd:
        sd1 = st.date_input("Start Date", value = date.today() - timedelta(days = 3680))
    with col_ed:
        ed1 = st.date_input("End Date", value = date.today(), min_value = sd1)


    months_diff2 = (ed1.year - sd1.year) * 12 + (ed1.month - sd1.month)
    st.caption(f"Selected time: {months_diff2} months")
    
    try:
        # Fetch historical data and create required data
        data_seasonality = yf.download([ticker1, ticker2], start=sd1, end=ed1 + timedelta(days=1))["Close"]
        data_seasonality = data_seasonality[[ticker1,ticker2]]
        data_seasonality['Price Ratio'] = data_seasonality[ticker1]/data_seasonality[ticker2]
        
        # make sure index is datetime for resample to work
        data_seasonality.index = pd.to_datetime(data_seasonality.index)
    
        #returns_seasonality = data_seasonality.pct_change().dropna()
        
        #st.dataframe(data_seasonality)

        # Compute monthly returns from price ratio
        monthly_returns = data_seasonality['Price Ratio'].resample('ME').last().pct_change().dropna()
        #st.dataframe(monthly_returns)

        # Put monthly returns into Year × Month format
        seasonality = monthly_returns.to_frame(name="Return")
        seasonality['Year'] = seasonality.index.year
        seasonality['Month'] = seasonality.index.month
        
        # Pivot table: rows = Year, cols = Month
        seasonality_table = seasonality.pivot(index="Year", columns="Month", values="Return")

        # Month order and labels
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        
        seasonality["MonthName"] = seasonality["Month"].map({
            1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching prices from Yahoo Finance…")
def fetch_prices(tickers: tuple, start: str, suffix: str) -> pd.DataFrame:
    syms = [t + suffix for t in tickers]
    raw = yf.download(syms, start=start, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(syms[0])
    close.columns = [c[: -len(suffix)] if suffix and c.endswith(suffix) else c for c in close.columns]
    return close.dropna(axis=1, how="all").sort_index()


def compute_returns(prices: pd.DataFrame, h: int, overlapping: bool) -> pd.DataFrame:
    if overlapping:
        return prices.pct_change(h, fill_method=None)
    return prices.iloc[::h].pct_change(fill_method=None)


def slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df.loc[start:end]


PRESETS = {
    "Since 1 Jan 2021": lambda ix: (pd.Timestamp(2021, 1, 1), ix[-1]),
    "Last 300 trading days": lambda ix: (ix[max(0, len(ix) - 300)], ix[-1]),
    "Last 1 year": lambda ix: (ix[-1] - pd.DateOffset(years=1), ix[-1]),
    "Last 3 years": lambda ix: (ix[-1] - pd.DateOffset(years=3), ix[-1]),
    "Last 5 years": lambda ix: (ix[-1] - pd.DateOffset(years=5), ix[-1]),
    "Year to date": lambda ix: (pd.Timestamp(ix[-1].year, 1, 1), ix[-1]),
    "Max available": lambda ix: (ix[0], ix[-1]),
}


def conditional_stats(R, conds, targets, mode, lag, thr, min_obs):
    masks = [OPS[op](R[t], v / 100) & R[t].notna() for t, op, v in conds]
    m = pd.concat(masks, axis=1)
    signal = m.all(axis=1) if mode == "ALL (AND)" else m.any(axis=1)

    tgt = R[targets].shift(-lag) if lag else R[targets]
    rows = []
    for t in targets:
        valid = tgt[t].notna()
        sig = signal & valid
        n = int(sig.sum())
        if n < min_obs:
            continue
        r_sig, r_all = tgt.loc[sig, t], tgt.loc[valid, t]
        p_c, p_b = (r_sig > thr).mean(), (r_all > thr).mean()
        up, dn = r_sig[r_sig > 0], r_sig[r_sig <= 0]
        p_up = len(up) / len(r_sig)
        avg_win = up.mean() if len(up) else 0.0
        avg_loss = dn.mean() if len(dn) else 0.0
        ev_c = p_up * avg_win + (1 - p_up) * avg_loss   # == mean return | condition
        ev_b = r_all.mean()
        rows.append({
            "Target": t,
            "Signals": n,
            "P(target > thr | cond) %": p_c * 100,
            "Base P %": p_b * 100,
            "Lift (pp)": (p_c - p_b) * 100,
            "P(up | cond) %": p_up * 100,
            "Avg win %": avg_win * 100,
            "Avg loss %": avg_loss * 100,
            "EV | cond %": ev_c * 100,
            "EV base %": ev_b * 100,
            "EV edge (pp)": (ev_c - ev_b) * 100,
            "Median ret | cond %": r_sig.median() * 100,
})
        
        # Base chart
        base = alt.Chart(seasonality).encode(
            x=alt.X("MonthName:O", sort=month_order, title="Month"),
            y=alt.Y("Year:O", sort="descending", title="Year")   # 👈 reverse order
        )
        
        # Heatmap
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "Return:Q",
                scale=alt.Scale(
                    domain=[-0.1, 0, 0.1],
                    range=["#b2182b", "#d9d9d9", "#14532d"]
                ),
                legend=None
            )
        )
        
        # White text with 1 decimal %
        text = base.mark_text(color="white").encode(
            text=alt.Text("Return:Q", format=".1%")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Lift (pp)", ascending=False).reset_index(drop=True)
    return out, signal


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    st.subheader("Return horizon")
    hz_label = st.radio(
        "Horizon", ["Weekly (5d)", "15 days", "Monthly (21d)", "Custom"], index=2,
        label_visibility="collapsed",
    )
    if hz_label == "Custom":
        horizon = int(st.number_input("Trading days", 1, 120, 10))
    else:
        horizon = HORIZONS[hz_label]
    overlapping = st.toggle(
        "Overlapping windows", value=True,
        help="On: rolling returns every day (more observations, but autocorrelated). "
             "Off: independent non-overlapping blocks.",
    )

    st.subheader("Timeline")
    tl_mode = st.radio(
        "Timeline mode", ["Quick preset", "Since date", "Date range", "Last N trading days"],
        label_visibility="collapsed",
    )
    if tl_mode == "Quick preset":
        preset = st.selectbox("Preset", list(PRESETS), index=0)
    elif tl_mode == "Since date":
        since = st.date_input("From", date(2021, 1, 1), min_value=date(2005, 1, 1), format="DD/MM/YYYY")
    elif tl_mode == "Date range":
        d1, d2 = st.columns(2)
        range_start = d1.date_input("From", date(2022, 1, 21), min_value=date(2005, 1, 1), format="DD/MM/YYYY")
        range_end = d2.date_input("To", date(2025, 7, 4), format="DD/MM/YYYY")
    else:
        n_days = int(st.number_input("N trading days", 30, 3000, 300, step=10))

    with st.expander("Universe & data source"):
        f_text = st.text_area("Factor ETFs", ", ".join(DEFAULT_FACTORS), height=90)
        s_text = st.text_area("Sector ETFs", ", ".join(DEFAULT_SECTORS), height=160)
        suffix = st.selectbox("Exchange suffix", [".NS", ".BO", ""], index=0)
        dl_start = st.date_input("Download history from", date(2015, 1, 1), format="DD/MM/YYYY")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

factor_list = parse_tickers(f_text)
sector_list = [t for t in parse_tickers(s_text) if t not in factor_list]

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
st.title("Factor & Sector Lab")
st.caption("Returns, relative performance and conditional probabilities across Indian factor and sector ETFs.")

all_tickers = tuple(factor_list + sector_list)
if not all_tickers:
    st.warning("Add at least one ticker in the sidebar.")
    st.stop()

try:
    prices = fetch_prices(all_tickers, str(dl_start), suffix)
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()

if prices.empty:
    st.error("No price data came back. Check the tickers and exchange suffix.")
    st.stop()

factors = prices[[c for c in factor_list if c in prices.columns]]
sectors = prices[[c for c in sector_list if c in prices.columns]]
missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)} (check the ticker symbols).")

ret_full = compute_returns(prices, horizon, overlapping)
ix = prices.index
if tl_mode == "Quick preset":
    w_start, w_end = PRESETS[preset](ix)
elif tl_mode == "Since date":
    w_start, w_end = pd.Timestamp(since), ix[-1]
elif tl_mode == "Date range":
    w_start, w_end = pd.Timestamp(range_start), pd.Timestamp(range_end)
else:
    w_start, w_end = ix[max(0, len(ix) - n_days)], ix[-1]

if w_start >= w_end:
    st.error("Start date must be before end date.")
    st.stop()

R = slice_window(ret_full, w_start, w_end)
px_win = slice_window(prices, w_start, w_end)
if px_win.empty:
    st.error("No data in the selected window. Try a different timeline.")
    st.stop()
if w_start < ix[0]:
    st.info(f"Data starts {ix[0]:%d %b %Y}; the window is clipped to that.")

st.caption(
    f"Window **{px_win.index[0]:%d %b %Y} → {px_win.index[-1]:%d %b %Y}** · "
    f"horizon **{horizon}d** · {'overlapping' if overlapping else 'non-overlapping'} returns"
)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_ov, tab_ex, tab_cp = st.tabs(["Overview", "Explore returns", "Conditional probability"])

# ---- Overview --------------------------------------------------------
with tab_ov:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "Factors", len(factors.columns)),
        (k2, "Sectors", len(sectors.columns)),
        (k3, "Window start", px_win.index[0].strftime("%d %b %Y") if len(px_win) else "—"),
        (k4, "Trading days in window", len(px_win)),
    ]:
        with col, st.container(border=True):
            st.metric(label, val)

    st.write("")
    st.subheader("Latest returns snapshot")
    st.caption("Trailing returns as of the most recent close, across all three horizons.")
    snap = pd.DataFrame({
        name: prices.pct_change(h, fill_method=None).iloc[-1] * 100 for name, h in HORIZONS.items()
    })

    def heat(df, title):
        fig = px.imshow(
            df, text_auto=".1f", aspect="auto",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            zmin=-10, zmax=10,
)
        
        # Combine
        chart = (heatmap + text).properties(
            width=600,
            height=400,
            title=f"Monthly Seasonality of {ticker1}/{ticker2}"
        fig.update_layout(
            title=title, coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0),
            height=max(160, 30 * len(df) + 90), xaxis_title=None, yaxis_title=None,
)
        fig.update_xaxes(side="top")
        return fig

        st.subheader("Seasonality Heatmap")
        st.altair_chart(chart, use_container_width=True)

        # RETURNS MATRIX
        # Create Year and Month columns
        returns_matrix = monthly_returns.to_frame("Return")
        returns_matrix["Year"] = returns_matrix.index.year
        returns_matrix["Month"] = returns_matrix.index.month_name().str[:3]  # Jan, Feb, ...
        
        # Pivot into matrix (Year = rows, Month = cols)
        returns_matrix = returns_matrix.pivot(index="Year", columns="Month", values="Return")
        
        # Reorder columns by calendar month
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul",
                       "Aug","Sep","Oct","Nov","Dec"]
        returns_matrix = returns_matrix.reindex(columns=month_order)
        
        #st.subheader("📅 Monthly Returns Matrix")
        #st.dataframe((returns_matrix*100).round(1).astype(str) + "%")

        # Clean matrix (remove rows with NaN)
        clean_matrix = returns_matrix.dropna(how="all")
        
        # % positive months (per column)
        pct_positive = ((clean_matrix > 0).sum() / clean_matrix.notnull().sum() * 100).round(0)
        
        # Avg returns per month
        avg_return = clean_matrix.mean()
        
        # Median returns per month
        median_return = clean_matrix.median()
        
        # Average of avg & median
        avg_median = (avg_return + median_return) / 2

        # Weighted average returns (latest years get higher weight)
        weighted_avg = {}
        for month in clean_matrix.columns:
            month_returns = clean_matrix[month].dropna()
            n = len(month_returns)
            if n > 0:
                weights = np.arange(1, n+1)  # 1...n
                weighted_avg[month] = np.average(month_returns.values, weights=weights)
            else:
                weighted_avg[month] = np.nan
        
        weighted_avg = pd.Series(weighted_avg)
        
        # Combine into dataframe
        stats = pd.DataFrame({
            "% Positive": pct_positive.astype(int).astype(str) + "%",
            "Avg Return": (avg_return*100).round(1).astype(str) + "%",
            "Median Return": (median_return*100).round(1).astype(str) + "%",
            "Avg of Avg & Median": (avg_median*100).round(1).astype(str) + "%",
            "Weighted Avg Return": (weighted_avg*100).round(1).astype(str) + "%"
        })
        
        # Reorder months
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        stats = stats.reindex(month_order)
        
        st.subheader("📊 Seasonality Stats by Month")
        st.dataframe(stats.T)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(heat(snap.loc[factors.columns], "Factors (%)"), use_container_width=True)
    with c2:
        st.plotly_chart(heat(snap.loc[sectors.columns], "Sectors (%)"), use_container_width=True)

# ---- Explore ---------------------------------------------------------
with tab_ex:
    default_sel = list(prices.columns[:4])
    sel = st.multiselect("Tickers", list(prices.columns), default=default_sel)
    if not sel:
        st.info("Pick at least one ticker.")
    else:
        left, right = st.columns(2)
        with left:
            rebased = px_win[sel].ffill()
            rebased = rebased / rebased.bfill().iloc[0] * 100
            fig = px.line(rebased, title="Rebased performance (start = 100)")
            fig.update_layout(legend_title=None, xaxis_title=None, yaxis_title=None,
                              margin=dict(l=0, r=0, t=40, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            long = R[sel].dropna(how="all").melt(var_name="Ticker", value_name="Return").dropna()
            long["Return"] *= 100
            fig = px.box(long, x="Ticker", y="Return", points=False,
                         title=f"Distribution of {horizon}d returns (%)")
            fig.add_hline(y=0, line_dash="dot", line_color="grey")
            fig.update_layout(xaxis_title=None, yaxis_title=None,
                              margin=dict(l=0, r=0, t=40, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{horizon}d return summary")
    summ = R.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summ[["mean", "std", "min", "50%", "max"]] *= 100
    summ["% positive"] = (R > 0).sum() / R.notna().sum() * 100
    summ.insert(0, "Group", ["Factor" if t in factors.columns else "Sector" for t in summ.index])
    summ = summ.rename(columns={"50%": "median", "count": "obs"})
    st.dataframe(
        summ, use_container_width=True,
        column_config={
            "obs": st.column_config.NumberColumn(format="%d"),
            "mean": st.column_config.NumberColumn("mean %", format="%.2f"),
            "std": st.column_config.NumberColumn("std %", format="%.2f"),
            "min": st.column_config.NumberColumn("min %", format="%.2f"),
            "median": st.column_config.NumberColumn("median %", format="%.2f"),
            "max": st.column_config.NumberColumn("max %", format="%.2f"),
            "% positive": st.column_config.ProgressColumn("% positive", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

# ---- Conditional probability ----------------------------------------
with tab_cp:
    st.subheader("When these move, what do the others do?")
    st.caption("Define conditions on any ETFs. Rows are combined with AND/OR, then each target is compared with its unconditional base rate.")

        
        # Check if data is empty (invalid ticker)
        if data_seasonality.empty or ticker1 not in data_seasonality.columns or ticker2 not in data_seasonality.columns:
            st.error("❗ Error: One or both tickers are invalid. Please enter correct stock/ETF symbols.")
            st.stop()  # Stop execution if tickers are invalid
    cols_avail = list(prices.columns)
    seed = [c for c in ["MOMOMENTUM", "MOVALUE"] if c in cols_avail] or cols_avail[:2]
    cond_df = pd.DataFrame({
        "Ticker": seed, "Operator": [">"] * len(seed), "Threshold %": [5.0] * len(seed),
    })

        # Check if the DataFrame is not empty and the index is within range
        if data_seasonality.empty:
            st.error("DataFrames are empty. Please check your data source.")
            st.stop()
        
        if len(data_seasonality[ticker1]) == 0:
            st.error(f"No data available for {ticker1}. Please check your data source.")
            st.stop()
    left, right = st.columns([1.3, 1])
    with left:
        cond_edit = st.data_editor(
            cond_df, num_rows="dynamic", use_container_width=True, key="conds",
            column_config={
                "Ticker": st.column_config.SelectboxColumn(options=cols_avail, required=True),
                "Operator": st.column_config.SelectboxColumn(options=list(OPS), required=True),
                "Threshold %": st.column_config.NumberColumn(step=0.5, format="%.1f", required=True),
            },
        )
    with right:
        mode = st.radio("Combine conditions", ["ALL (AND)", "ANY (OR)"], horizontal=True)
        tgt_group = st.radio("Targets", ["Sectors", "Factors", "All others"], horizontal=True)
        o1, o2, o3 = st.columns(3)
        thr_pct = o1.number_input("Target thr %", value=0.0, step=0.5)
        lag = o2.number_input("Lag (rows)", 0, 250, 0,
                              help="0 = same period. k = target's return k rows later (forward-looking).")
        min_obs = o3.number_input("Min signals", 1, 500, 10)

        #returns_seasonality = returns_seasonality.to_frame()
    conds = [
        (r["Ticker"], r["Operator"], float(r["Threshold %"]))
        for _, r in cond_edit.dropna().iterrows()
        if r["Ticker"] in cols_avail
    ]
    if not conds:
        st.info("Add at least one condition above.")
        st.stop()

        #st.dataframe(returns_seasonality)
        
    except Exception as e:
        st.error(f"❗ Error fetching historical data: {e}")
        st.stop()  # Stops execution immediately after showing error 
    cond_set = {c[0] for c in conds}
    pool = {"Sectors": sectors.columns, "Factors": factors.columns, "All others": prices.columns}[tgt_group]
    targets = [t for t in pool if t not in cond_set]

    res, signal = conditional_stats(R, conds, targets, mode, int(lag), thr_pct / 100, int(min_obs))

    m1, m2, m3 = st.columns(3)
    with m1, st.container(border=True):
        st.metric("Signal periods", int(signal.sum()))
    with m2, st.container(border=True):
        st.metric("Share of window", f"{signal.mean() * 100:.1f}%")
    with m3, st.container(border=True):
        st.metric("Horizon / lag", f"{horizon}d / {int(lag)}")

    if overlapping and signal.sum() > 0:
        st.caption("⚠️ Overlapping windows are autocorrelated, so the effective sample is smaller than the signal count. "
                   "Switch off overlap in the sidebar for independent observations.")

    if res.empty:
        st.warning("No target has enough signal observations. Lower the thresholds, shorten the lag, or reduce Min signals.")
    else:
        rank_by = st.selectbox("Rank & chart by", ["EV edge (pp)", "EV | cond %", "Lift (pp)"], index=0)
        res = res.sort_values(rank_by, ascending=False).reset_index(drop=True)
        res["Direction"] = np.where(res[rank_by] >= 0, "Positive", "Negative")
        fig = px.bar(
            res.sort_values(rank_by), x=rank_by, y="Target", orientation="h",
            color="Direction", color_discrete_map={"Positive": "#2e9e5b", "Negative": "#c8473d"},
            hover_data={"Signals": True, "Direction": False},
            title=f"{rank_by} by target (condition vs unconditional)",
        )
        fig.update_layout(showlegend=False, yaxis_title=None,
                          margin=dict(l=0, r=0, t=40, b=0), height=max(280, 26 * len(res) + 90))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("**EV | cond** = P(up)·avg win + P(down)·avg loss, i.e. the mean target return when the condition fires. "
                   "**EV edge** = EV | cond − unconditional EV.")
        order = ["Target", "Signals", "P(target > thr | cond) %", "Base P %", "Lift (pp)", "P(up | cond) %",
                 "Avg win %", "Avg loss %", "EV | cond %", "EV base %", "EV edge (pp)", "Median ret | cond %"]
        st.dataframe(
            res[order], use_container_width=True, hide_index=True,
            column_config={
                "Signals": st.column_config.NumberColumn(format="%d"),
                "P(target > thr | cond) %": st.column_config.ProgressColumn(
                    "P(tgt > thr | cond)", min_value=0, max_value=100, format="%.0f%%"),
                "Base P %": st.column_config.NumberColumn("Base P", format="%.0f%%"),
                "Lift (pp)": st.column_config.NumberColumn("Lift", format="%+.1f pp"),
                "P(up | cond) %": st.column_config.ProgressColumn(
                    "P(up | cond)", min_value=0, max_value=100, format="%.0f%%"),
                "Avg win %": st.column_config.NumberColumn("Avg win", format="%+.2f%%"),
                "Avg loss %": st.column_config.NumberColumn("Avg loss", format="%+.2f%%"),
                "EV | cond %": st.column_config.NumberColumn("EV | cond", format="%+.2f%%"),
                "EV base %": st.column_config.NumberColumn("EV base", format="%+.2f%%"),
                "EV edge (pp)": st.column_config.NumberColumn("EV edge", format="%+.2f pp"),
                "Median ret | cond %": st.column_config.NumberColumn("Median | cond", format="%+.2f%%"),
            },
        )

    with st.expander("Dates when the condition fired"):
        fired = R.loc[signal, [c[0] for c in conds]] * 100
        st.dataframe(fired.round(2), use_container_width=True)
