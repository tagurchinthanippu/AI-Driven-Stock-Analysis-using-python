import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_searchbox import st_searchbox

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro-Quant AI Terminal", layout="wide")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SEARCH ENGINE ---
def search_symbols(searchterm: str):
    """Dynamic suggestions for the search box."""
    popular = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "BTC-USD", "ETH-USD", "RELIANCE.NS", "TCS.NS"]
    if not searchterm:
        return popular
    return [s for s in popular if searchterm.upper() in s]

# --- DATA ENGINE (FIXED FOR GRAPHS) ---
@st.cache_data(ttl=3600)
def load_data(symbol, days):
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        # multi_level_index=False is CRITICAL for newer yfinance versions
        df = yf.download(symbol, start=start, end=end, interval="1h", multi_level_index=False)
        if df.empty: return None
        # Clean column names to ensure they are simple strings
        df.columns = [str(col) for col in df.columns]
        return df
    except Exception:
        return None

# --- FEATURE ENGINEERING ---
def add_features(data):
    df = data.copy()
    if len(df) < 30: return pd.DataFrame() # Need enough data for indicators
    
    # 1. Price Momentum
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Moving Average (Trend)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 3. Volatility (Risk)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 4. RSI (Strength)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # 5. Target (The 'Next-Hour' move)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

# --- APP LAYOUT ---
st.title("🎯 Pro-Quant AI Terminal")
st.caption("Real-time Machine Learning Market Analysis")

with st.sidebar:
    st.header("🔍 Configuration")
    # THE DYNAMIC SEARCHBOX
    ticker = st_searchbox(
        search_symbols,
        placeholder="Search (e.g. TSLA, BTC-USD)...",
        key="symbol_search",
        default="AAPL"
    )
    if ticker: ticker = ticker.upper()
    
    st.divider()
    days_back = st.slider("Historical Horizon (Days)", 30, 365, 90)
    threshold = st.slider("AI Confidence Threshold", 0.5, 0.9, 0.65)
    st.info("The threshold filters low-confidence signals. Set higher for safer trades.")

if ticker:
    df_raw = load_data(ticker, days_back)
    
    if df_raw is None or df_raw.empty:
        st.error(f"⚠️ No data found for {ticker}. Try a different symbol.")
    else:
        df = add_features(df_raw)
        
        if df.empty:
            st.warning("Not enough historical data to generate indicators.")
        else:
            # --- AI MODELING ---
            features = ['Close', 'Volatility', 'RSI', 'SMA_20']
            X = df[features]
            y = df['Target']

            # Split 80/20 for Backtesting
            split = int(len(X) * 0.8)
            model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(X.iloc[:split], y.iloc[:split])

            # Latest Prediction
            latest_data = X.iloc[-1:]
            prob = model.predict_proba(latest_data)[0][1]
            signal = "BULLISH 📈" if prob > threshold else "NEUTRAL/BEARISH 📉"

            # --- UI: TOP METRICS ---
            m1, m2, m3 = st.columns(3)
            current_price = df['Close'].iloc[-1]
            price_change = current_price - df['Close'].iloc[-2]
            
            m1.metric("Current Price", f"${current_price:,.2f}", f"{price_change:.2f}")
            m2.metric("AI Signal", signal)
            m3.metric("AI Confidence", f"{prob*100:.1f}%")

            # --- UI: CHART ---
            st.subheader(f"📊 {ticker} Analysis Chart")
            fig = go.Figure()
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], 
                low=df['Low'], close=df['Close'], name="OHLC"
            ))
            # Trend Line
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name="SMA 20"))
            
            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- UI: BACKTEST ---
            st.subheader("🏁 Performance Check (Out-of-Sample)")
            y_test_probs = model.predict_proba(X.iloc[split:])[:, 1]
            y_test_preds = (y_test_probs > threshold).astype(int)
            
            returns = df['Returns'].iloc[split:]
            strat_returns = (1 + (y_test_preds * returns)).cumprod()
            mkt_returns = (1 + returns).cumprod()
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**AI Strategy Return**")
                st.title(f"{(strat_returns.iloc[-1]-1)*100:.2f}%")
            with c2:
                st.write("**Buy & Hold Return**")
                st.title(f"{(mkt_returns.iloc[-1]-1)*100:.2f}%")

            st.caption("Strategy Return assumes entering a trade only when AI confidence exceeds your threshold.")