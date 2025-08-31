# Libraries
# ------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import the statsmodels module for regression and the adfuller function
# Import statsmodels.formula.api
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from datetime import date, timedelta

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
    default_start = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
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
    
        if submit and ticker1 and ticker2:
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
        monthly_returns = data_seasonality['Price Ratio'].resample('M').last().pct_change().dropna()
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
        )
        
        # Combine
        chart = (heatmap + text).properties(
            width=600,
            height=400,
            title="Monthly Seasonality of Price Ratio"
        )
        
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

        # Drop NaN years (clean matrix)
        clean_matrix = returns_matrix.dropna(how="all")
        
        # Stats
        positive_pct = ((clean_matrix > 0).sum() / clean_matrix.notnull().sum() * 100).round(0)
        avg_returns = clean_matrix.mean().round(2)
        median_returns = clean_matrix.median().round(2)
        avg_of_avg_median = ((avg_returns + median_returns) / 2).round(2)
        
        # Combine stats into DataFrame
        stats_df = pd.DataFrame({
            "% Positive Months": positive_pct,
            "Average Return": avg_returns,
            "Median Return": median_returns,
            "Avg of Avg & Median": avg_of_avg_median
        }).T   # 👈 transpose so rows = stats
        
        st.subheader("📊 Seasonality Stats (Transposed)")
        st.dataframe(stats_df)


        
        # Check if data is empty (invalid ticker)
        if data_seasonality.empty or ticker1 not in data_seasonality.columns or ticker2 not in data_seasonality.columns:
            st.error("❗ Error: One or both tickers are invalid. Please enter correct stock/ETF symbols.")
            st.stop()  # Stop execution if tickers are invalid

        # Check if the DataFrame is not empty and the index is within range
        if data_seasonality.empty:
            st.error("DataFrames are empty. Please check your data source.")
            st.stop()
        
        if len(data_seasonality[ticker1]) == 0:
            st.error(f"No data available for {ticker1}. Please check your data source.")
            st.stop()

        #returns_seasonality = returns_seasonality.to_frame()

        #st.dataframe(returns_seasonality)
        
    except Exception as e:
        st.error(f"❗ Error fetching historical data: {e}")
        st.stop()  # Stops execution immediately after showing error 


