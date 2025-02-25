from flask import Flask, request, jsonify
import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pandas_ta as ta
import numpy as np
import logging
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import io, base64
import warnings
warnings.filterwarnings("ignore")
# Set timezone for India region
india_tz = pytz.timezone('Asia/Kolkata')

# Configure logging to file
logging.basicConfig(filename='trades.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

app = Flask(__name__)

# ----- OOP Classes -----
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

@app.route('/trade', methods=['POST'])
def trade():
    content = request.json
    symbol = content.get('symbol')
    username = content.get('username', 'default_user')
    rsi_period = int(content.get('rsi_period', 14))
    buy_threshold = float(content.get('buy_threshold', 30))
    sell_threshold = float(content.get('sell_threshold', 70))
    trade_type = content.get('trade_type', 'long')
    strategy = content.get('strategy', 'RSI')
    ma_period = int(content.get('ma_period', 20))
    timeframe = content.get('timeframe', '1m')
    
    # Validate strategy input - if RSI_MA is requested, default to RSI
    if strategy.upper() == "RSI_MA":
        strategy = "RSI"
    
    user = User(username, rsi_period, buy_threshold, sell_threshold, trade_type, strategy, ma_period)
    strategy_obj = TradingStrategy(user)
    result = strategy_obj.execute_strategy(symbol, timeframe)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)