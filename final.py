import streamlit as st
import pandas as pd
import datetime
import ccxt
import pandas_ta as ta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import warnings
from datetime import datetime
import pytz
import logging
import threading
import time
warnings.filterwarnings("ignore")

# Set timezone for India region
india_tz = pytz.timezone('Asia/Kolkata')

# Configure logging to file
logging.basicConfig(filename='trades.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# ----- OOP Classes for Backend -----
class User:
    def __init__(self, username, rsi_period=14, buy_threshold=30, sell_threshold=70, trade_type='long', strategy='RSI', ma_period=20):
        self.username = username
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trade_type = trade_type.lower()      # "long" or "short"
        self.strategy = strategy.upper()          # "RSI" or "MA"
        self.ma_period = ma_period

class TradeLogger:
    @staticmethod
    def log_trade(trade_details):
        logging.info(trade_details)

class TradingStrategy:
    def __init__(self, user: User):
        self.user = user
        self.binance = ccxt.binance({
            'options': {
                'defaultType': 'spot'
            }
        })
    
    def fetch_data(self, symbol, timeframe='1m', limit=100):
        """Fetch historical OHLCV data from Binance."""
        try:
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')\
                                .dt.tz_localize('UTC').dt.tz_convert(india_tz)
            data.set_index('datetime', inplace=True)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_rsi(self, data):
        data['RSI'] = ta.rsi(data['close'], length=self.user.rsi_period, mamode='wilder')
        data = data.iloc[self.user.rsi_period:]
        return data
    
    def generate_plots(self, data, trades):
        """Generate a plot of close prices, RSI (and MA if available) with trade markers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(data.index, data['close'], label='Close Price', color='blue')
        ax1.set_ylabel("Price", color='blue')
        ax1.set_title(f"{self.user.strategy} Strategy - {self.user.trade_type.capitalize()} Trades (RSI Period: {self.user.rsi_period}, MA Period: {self.user.ma_period})")
        
        # Plot trade markers
        for trade in trades:
            try:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                if self.user.trade_type == 'long':
                    entry_marker, exit_marker = '^', 'v'
                    entry_color, exit_color = 'green', 'red'
                else:
                    entry_marker, exit_marker = 'v', '^'
                    entry_color, exit_color = 'red', 'green'
                ax1.plot(entry_time, entry_price, marker=entry_marker, markersize=10, color=entry_color)
                ax1.plot(exit_time, exit_price, marker=exit_marker, markersize=10, color=exit_color)
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 'k--', alpha=0.3)
                if self.user.trade_type == 'long':
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl = ((entry_price - exit_price) / entry_price) * 100
                midpoint_time = entry_time + (exit_time - entry_time) / 2
                midpoint_price = (entry_price + exit_price) / 2
                ax1.annotate(f"{pnl:.2f}%", (midpoint_time, midpoint_price), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
            except Exception as e:
                print(f"Error plotting trade: {e}")
                continue
        
        # Plot RSI
        ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
        ax2.axhline(y=self.user.buy_threshold, color='green', linestyle='--', label=f'Buy Threshold ({self.user.buy_threshold})')
        ax2.axhline(y=self.user.sell_threshold, color='red', linestyle='--', label=f'Sell Threshold ({self.user.sell_threshold})')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 100)
        ax2.legend()
        
        # If MA is available, plot it on the price chart
        if self.user.strategy == "MA" and 'MA' in data.columns:
            ax1.plot(data.index, data['MA'], label=f"MA ({self.user.ma_period})", color='magenta', linestyle='--')
            ax1.legend()
        
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    def execute_strategy(self, symbol, timeframe='1m'):
        """Execute the chosen strategy."""
        data = self.fetch_data(symbol, timeframe=timeframe, limit=1000)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data = self.calculate_rsi(data)
        
        # For MA strategy, compute the moving average
        if self.user.strategy == "MA":
            data['MA'] = data['close'].rolling(window=self.user.ma_period).mean()
            data = data.iloc[self.user.ma_period-1:]
        
        trades = []
        open_positions = []
        
        # ----- Strategy Implementation -----
        if self.user.strategy == "RSI":
            for i in range(1, len(data)):
                prev_rsi = data['RSI'].iloc[i-1]
                curr_rsi = data['RSI'].iloc[i]
                current_time = data.index[i]
                current_price = data['close'].iloc[i]
                if self.user.trade_type == 'long':
                    if prev_rsi < self.user.buy_threshold and curr_rsi >= self.user.buy_threshold:
                        if not open_positions:
                            open_positions.append({'entry_time': current_time, 'entry_price': current_price, 'entry_rsi': curr_rsi})
                    elif prev_rsi > self.user.sell_threshold and curr_rsi <= self.user.sell_threshold:
                        for pos in open_positions[:]:
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],
                                'exit_time': str(current_time),
                                'exit_price': current_price,
                                'exit_rsi': curr_rsi,
                                'trade_type': 'long',
                                'profit_pct': ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(f"Long RSI trade for {symbol}: Buy at {pos['entry_time']} (price: {pos['entry_price']}, RSI: {pos['entry_rsi']}) | Sell at {current_time} (price: {current_price}, RSI: {curr_rsi}) | P/L: {trade['profit_pct']:.2f}%")
                else:
                    if prev_rsi > self.user.sell_threshold and curr_rsi <= self.user.sell_threshold:
                        if not open_positions:
                            open_positions.append({'entry_time': current_time, 'entry_price': current_price, 'entry_rsi': curr_rsi})
                    elif prev_rsi < self.user.buy_threshold and curr_rsi >= self.user.buy_threshold:
                        for pos in open_positions[:]:
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],
                                'exit_time': str(current_time),
                                'exit_price': current_price,
                                'exit_rsi': curr_rsi,
                                'trade_type': 'short',
                                'profit_pct': ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(f"Short RSI trade for {symbol}: Sell at {pos['entry_time']} (price: {pos['entry_price']}, RSI: {pos['entry_rsi']}) | Cover at {current_time} (price: {current_price}, RSI: {curr_rsi}) | P/L: {trade['profit_pct']:.2f}%")
        elif self.user.strategy == "MA":
            for i in range(1, len(data)):
                prev_price = data['close'].iloc[i-1]
                curr_price = data['close'].iloc[i]
                prev_ma = data['MA'].iloc[i-1]
                curr_ma = data['MA'].iloc[i]
                current_time = data.index[i]
                if self.user.trade_type == 'long':
                    if prev_price < prev_ma and curr_price >= curr_ma:
                        if not open_positions:
                            open_positions.append({
                                'entry_time': current_time,
                                'entry_price': curr_price,
                                'entry_rsi': None  # Placeholder for MA strategy
                            })
                    elif prev_price > prev_ma and curr_price <= curr_ma:
                        for pos in open_positions[:]:
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],  # Will be None
                                'exit_time': str(current_time),
                                'exit_price': curr_price,
                                'exit_rsi': None,  # Placeholder for MA strategy
                                'trade_type': 'long',
                                'profit_pct': ((curr_price - pos['entry_price']) / pos['entry_price']) * 100
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(
                                f"Long MA trade for {symbol}: Buy at {pos['entry_time']} (price: {pos['entry_price']}) | Sell at {current_time} (price: {curr_price}) | P/L: {trade['profit_pct']:.2f}%"
                            )
                else:  # Short trades
                    if prev_price > prev_ma and curr_price <= curr_ma:
                        if not open_positions:
                            open_positions.append({
                                'entry_time': current_time,
                                'entry_price': curr_price,
                                'entry_rsi': None  # Placeholder for MA strategy
                            })
                    elif prev_price < prev_ma and curr_price >= curr_ma:
                        for pos in open_positions[:]:
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],  # Will be None
                                'exit_time': str(current_time),
                                'exit_price': curr_price,
                                'exit_rsi': None,  # Placeholder for MA strategy
                                'trade_type': 'short',
                                'profit_pct': ((pos['entry_price'] - curr_price) / pos['entry_price']) * 100
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(
                                f"Short MA trade for {symbol}: Sell at {pos['entry_time']} (price: {pos['entry_price']}) | Cover at {current_time} (price: {curr_price}) | P/L: {trade['profit_pct']:.2f}%"
                            )
        
        plot_image = self.generate_plots(data, trades)
        trade_summary = {
            "total_trades": len(trades),
            "winning_trades": sum(1 for t in trades if t.get('profit_pct', 0) > 0),
            "total_profit_pct": sum(t.get('profit_pct', 0) for t in trades),
            "avg_profit_per_trade": sum(t.get('profit_pct', 0) for t in trades) / len(trades) if trades else 0
        }
        df = data.reset_index()
        df['datetime'] = df['datetime'].astype(str)
        data_json = df.to_dict('records')
        return {
            "trades": trades, 
            "data": data_json, 
            "plot": plot_image,
            "summary": trade_summary
        }

# ----- Streamlit Frontend -----
st.title("📈 Trading Strategy Backtester")

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
        timeframe = st.sidebar.selectbox("Timeframe", options=["1m", "5m", "15m", "1h", "4h", "1d"],
                                         help="Candle timeframe for analysis.")
        submit_button = st.sidebar.button("Run Backtest", type="primary")
        
    with col2:
        if submit_button:
            with st.spinner("Running backtest..."):
                try:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Create user and strategy objects
                    user = User(
                        username=username,
                        rsi_period=rsi_period,
                        buy_threshold=buy_threshold,
                        sell_threshold=sell_threshold,
                        trade_type=trade_type,
                        strategy=strategy,
                        ma_period=ma_period
                    )
                    
                    progress_bar.progress(10, text="Initializing strategy...")
                    
                    strategy_obj = TradingStrategy(user)
                    
                    progress_bar.progress(30, text="Fetching market data...")
                    
                    # Execute strategy directly (no HTTP request needed since we're in the same app)
                    result = strategy_obj.execute_strategy(symbol, timeframe)
                    
                    progress_bar.progress(90, text="Processing results...")
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        trades = result.get("trades", [])
                        historical_data = result.get("data", [])
                        plot_img = result.get("plot", "")
                        summary = result.get("summary", {})
                        
                        st.subheader("Strategy Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Trades", summary.get("total_trades", 0))
                        with col2:
                            win_pct = 0
                            if summary.get("total_trades", 0) > 0:
                                win_pct = int(summary.get("winning_trades", 0) / summary.get("total_trades") * 100)
                            st.metric("Winning Trades", f"{summary.get('winning_trades', 0)} ({win_pct}%)")
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
                            # Only include columns that exist in the dataframe
                            display_columns = [col for col in display_columns if col in trade_df.columns]
                            trade_df_display = trade_df[display_columns].copy()
                            if 'profit_pct' in trade_df_display.columns:
                                trade_df_display['profit_pct'] = trade_df_display['profit_pct'].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(trade_df_display, use_container_width=True)
                        else:
                            st.info("No trades executed. Adjust thresholds or strategy parameters.")
                        
                        if trades:
                            csv = trade_df.to_csv(index=False)
                            st.download_button(label="Download Trade Data (CSV)",
                                               data=csv,
                                               file_name=f"trades_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                                               mime="text/csv")
                        
                        with st.expander("Historical Price and RSI Data"):
                            if historical_data:
                                hist_df = pd.DataFrame(historical_data)
                                st.dataframe(hist_df)
                            else:
                                st.info("No historical data available.")
                    
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.5)  # Let the user see the 100% progress
                    progress_bar.empty()
                    
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
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
    st.markdown("""
    ### Trading Strategy Backtester
    
    This application allows you to backtest trading strategies on cryptocurrency pairs using historical data from Binance.
    
    #### Available Strategies
    
    1. **RSI (Relative Strength Index)**
       - A momentum oscillator that measures the speed and change of price movements
       - Values range from 0 to 100
       - Traditional interpretation: RSI > 70 = overbought, RSI < 30 = oversold
       - Adjust thresholds according to market conditions
    
    2. **MA (Moving Average)**
       - Price crossing above/below its moving average generates signals
       - Shorter periods are more responsive but may generate more false signals
       - Longer periods are slower but may miss profit opportunities
    
    #### Trade Types
    
    - **Long (Buy-Sell)**: Profit from upward price movements
    - **Short (Sell-Buy)**: Profit from downward price movements
    
    #### Parameters
    
    - **RSI Period**: Number of candles to calculate RSI (default: 14)
    - **Buy/Cover Threshold**: RSI level to trigger buy (long) or cover (short) signals
    - **Sell/Short Threshold**: RSI level to trigger sell (long) or short (short) signals
    - **MA Period**: Number of candles to calculate the moving average
    
    #### Limitations
    
    - Past performance does not guarantee future results
    - No transaction fees or slippage included in calculations
    - Data is fetched from Binance public API and may have limitations
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
