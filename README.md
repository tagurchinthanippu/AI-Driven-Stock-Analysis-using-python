# Pro-Quant AI Terminal 💹

This repository contains a high-performance **Quant-Trading Dashboard** built with Python. It bridges the gap between raw financial data and machine learning, using an **XGBoost Classifier** to predict short-term price movements (hourly candles) for any ticker available on Yahoo Finance.

---

## 🚀 Features

* **Live Data Integration:** Fetches real-time market data via `yfinance`.
* **Feature Engineering:** Automatically calculates **RSI** (Momentum), **Volatility** (Risk), and **SMA** (Trend) to feed the AI.
* **XGBoost Intelligence:** Uses a state-of-the-art gradient boosting model to calculate the probability of a "Bullish" or "Bearish" next move.
* **Interactive Visualization:** Dynamic candlestick charts powered by **Plotly**.
* **Backtesting Engine:** Compares the AI strategy's historical performance against a standard "Buy & Hold" benchmark.
* **Adjustable Thresholds:** Users can tune the "Confidence Threshold" to filter out low-probability signals.

---

## 🛠️ Installation & Usage

Follow these steps to get your AI trading terminal up and running:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pro-quant-terminal.git
cd pro-quant-terminal
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install streamlit yfinance pandas numpy xgboost plotly
```

### 3. Run the Application
```bash
streamlit run stock_app.py
```

---

## 📂 Project Structure

* `stock_app.py`: The main Streamlit application containing the UI, data processing, and ML logic.
* **Data Engine:** Handles hourly data retrieval and caching for performance.
* **ML Pipeline:** Trains an XGBoost model on the first 80% of data and tests on the remaining 20%.

---

## ⚠️ Disclaimer

**Not Financial Advice:** This tool is for educational and research purposes only. Trading stocks and cryptocurrencies involves significant risk. Never trade with money you cannot afford to lose. The AI model's accuracy is based on historical patterns and does not guarantee future results.

---

## 🤝 Contributing
Feel free to fork this project, open issues, or submit pull requests to add new technical indicators or improve the prediction model!

---

**Would you like me to generate a `requirements.txt` file for you to make the installation even easier?**
