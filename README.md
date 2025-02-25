# 📈 Trading Strategy Backtester

A **Streamlit-based Trading Strategy Backtester** using **RSI (Relative Strength Index)** and **Moving Averages (MA)** to simulate and analyze trades on Binance.

## 🚀 Features
- Fetches real-time **OHLCV data** from Binance
- Implements **RSI and Moving Average-based strategies**
- **Visualizes trade entries & exits** with Matplotlib
- **Logs trades** for later analysis
- **Interactive parameters** to tweak strategy settings

---

## 📌 Strategy Overview
### 1️⃣ RSI-Based Strategy
The **RSI strategy** takes trades based on RSI values:
- **Buy** when RSI crosses above the lower threshold (e.g., 30)
- **Sell** when RSI crosses below the upper threshold (e.g., 70)

![RSI Strategy](RSI_strategy.png)

#### 📊 RSI Graph Example
![RSI Graph](RSI_graph.png)

---

### 2️⃣ Moving Average (MA) Strategy
The **MA strategy** takes trades based on price and moving averages:
- **Buy** when price crosses above the MA
- **Sell** when price crosses below the MA

![MA Parameters](MA_parametres.png)

#### 📈 MA Graph Example
![MA Graph](MA_graph.png)

---

## 📜 Trade Details Feature
This feature logs all trade entries and exits for analysis.

![Trade Details](trade_details_feature.png)

## 📁 Logs
Trade details and executions are logged for further review.

![Logs](Logs.png)

---

## 🔧 Installation & Setup
Follow these steps to run the project locally.

### 1️⃣ Clone the Repository
