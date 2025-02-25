# finfrontend.py
import streamlit as st
import requests
import pandas as pd
import datetime

st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trading Strategy Backtester")
st.markdown("Backtest your trading strategies using historical data.")

tab1, tab2, tab3 = st.tabs(["Trading Strategy", "Documentation", "About"])

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.sidebar.header("💼 User Settings")
        with st.sidebar.expander("User Profile", expanded=True):
            username = st.text_input("Username", "default_user")
        with st.sidebar.expander("RSI Parameters", expanded=True):
            rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14, step=1,
                                         help="Number of periods for RSI calculation.")
            buy_threshold = st.number_input("Buy/Cover Threshold", min_value=1.0, max_value=99.0, value=30.0, step=1.0,
                                            help="RSI level to trigger buy (long) or cover (short) signal.")
            sell_threshold = st.number_input("Sell/Short Threshold", min_value=1.0, max_value=99.0, value=70.0, step=1.0,
                                             help="RSI level to trigger sell (long) or short (short) signal.")
        with st.sidebar.expander("Moving Average Parameters", expanded=True):
            ma_period = st.number_input("MA Period", min_value=2, max_value=100, value=20, step=1,
                                        help="Number of periods for the moving average.")
        with st.sidebar.expander("Trading Parameters", expanded=True):
            trade_type = st.selectbox("Trade Type", options=["long", "short"],
                                      help="Choose whether to simulate long or short trades.")
            strategy = st.selectbox("Strategy", options=["RSI", "MA"],
                                    help="RSI: RSI-based signals | MA: Price crossover of MA")
        st.sidebar.markdown("---")
        symbol = st.sidebar.selectbox("Select Symbol", options=["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"],
                                      help="Cryptocurrency trading pair.")
        submit_button = st.sidebar.button("Run Backtest", type="primary")
        
    with col2:
        if submit_button:
            with st.spinner("Running backtest..."):
                payload = {
                    "username": username,
                    "symbol": symbol,
                    "rsi_period": rsi_period,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "trade_type": trade_type,
                    "strategy": strategy,
                    "ma_period": ma_period,
                    "timeframe": "1m"
                }
                try:
                    response = requests.post("http://127.0.0.1:5000/trade", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        trades = result.get("trades", [])
                        historical_data = result.get("data", [])
                        plot_img = result.get("plot", "")
                        summary = result.get("summary", {})
                        
                        st.subheader("Strategy Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Trades", summary.get("total_trades", 0))
                        with col2:
                            st.metric("Winning Trades", f"{summary.get('winning_trades', 0)} ({int(summary.get('winning_trades', 0)/max(1, summary.get('total_trades', 1))*100)}%)")
                        with col3:
                            st.metric("Total Return", f"{summary.get('total_profit_pct', 0):.2f}%")
                        with col4:
                            st.metric("Avg. Trade", f"{summary.get('avg_profit_per_trade', 0):.2f}%")
                        
                        st.subheader("Technical Analysis Plot")
                        if plot_img:
                            st.image("data:image/png;base64," + plot_img, use_container_width=True)
                        else:
                            st.info("No plot available.")
                        
                        st.subheader("Trade Details")
                        if trades:
                            trade_df = pd.DataFrame(trades)
                            display_columns = ['symbol', 'trade_type', 'entry_time', 'entry_price', 'entry_rsi', 'exit_time', 'exit_price', 'exit_rsi', 'profit_pct']
                            trade_df_display = trade_df[display_columns].copy()
                            trade_df_display['profit_pct'] = trade_df_display['profit_pct'].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(trade_df_display, use_container_width=True)
                        else:
                            st.info("No trades executed. Adjust thresholds or strategy parameters.")
                        
                        if trades:
                            csv = trade_df.to_csv(index=False)
                            st.download_button(label="Download Trade Data (CSV)",
                                               data=csv,
                                               file_name=f"trades_{symbol.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                               mime="text/csv")
                        
                        with st.expander("Historical Price and RSI Data"):
                            if historical_data:
                                hist_df = pd.DataFrame(historical_data)
                                st.dataframe(hist_df)
                            else:
                                st.info("No historical data available.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    st.info("Ensure the backend server is running on http://127.0.0.1:5000")
        else:
            st.info("👈 Configure your strategy parameters and click 'Run Backtest'")
            st.subheader("How these strategies work")
            st.markdown("""
            **RSI Strategy:**
            - *Long:* Buy when RSI crosses above the buy threshold; sell when RSI crosses below the sell threshold.
            - *Short:* Sell when RSI crosses below the sell threshold; cover when RSI crosses above the buy threshold.
            
            **MA Strategy:**
            - *Long:* Buy when the price crosses above its moving average; sell when it crosses below.
            - *Short:* Sell when the price crosses below its moving average; cover when it crosses above.
            """)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/RSI_example.png/1200px-RSI_example.png",
                     caption="Example RSI Chart", use_container_width=True)

with tab2:
    st.header("Documentation")
    st.subheader("Technical Indicators and Strategies")
    st.markdown("""
    **Relative Strength Index (RSI):**
    - Measures momentum and oscillates between 0 and 100.
    - Common thresholds: overbought >70, oversold <30.
    
    **Moving Average (MA):**
    - A Simple Moving Average (SMA) calculated over a specified period.
    - Helps identify trends and potential support/resistance levels.
    
    **Strategies:**
    - **RSI:** Based on RSI crossovers.
    - **MA:** Based on price crossing the moving average.
    
    
    **Data Source:**
    - You can choose to use live data from Binance or load a CSV dataset of minute-wise data.
    """)

    
with tab3:
    st.header("About")
    st.markdown("""
    This tool was developed as a proof-of-concept for backtesting multiple trading strategies using technical indicators.
    
    **Features:**
    - Supports RSI and MA strategies.
    - Configurable parameters for RSI and MA.
    - Option to load minute-wise CSV data.
    - Visualizations of price, RSI, moving averages, and trade signals.

    """)
