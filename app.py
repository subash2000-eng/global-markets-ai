import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Global Markets AI",
    page_icon="📈",
    layout="wide"
)
# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* MAIN TITLE */

.main-title {
    text-align:center;
    font-size:70px;
    font-weight:800;
    color:white;
    margin-top:10px;
}

.sub-title {
    text-align:center;
    font-size:20px;
    color:#94a3b8;
    margin-bottom:30px;
}

/* GLASS CARD */

.glass {
    background: #111827;
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 25px;
    border:1px solid rgba(255,255,255,0.12);
    box-shadow:0px 8px 30px rgba(0,0,0,0.3);
}

/* METRIC CARDS */

.metric-card {
    background: #111827;
    backdrop-filter: blur(12px);
    border-radius:20px;
    padding:20px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.1);
    transition:0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

/* SEARCH BOX */

.search-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius:20px;
    padding:25px;
    margin-bottom:20px;
    border:1px solid rgba(255,255,255,0.1);
}

/* INPUT TEXT */

.stTextInput input {

    color: white !important;

    background-color: #111827 !important;

    border: 1px solid rgba(255,255,255,0.2) !important;

    border-radius: 12px !important;

    padding: 12px !important;
}

/* PLACEHOLDER */

.stTextInput input::placeholder {

    color: #cbd5e1 !important;

    opacity: 1;
}

/* LABEL */

.stTextInput label {

    color: white !important;

    font-weight: 600 !important;
}

/* BUTTON */

.stButton button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #3b82f6
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 14px !important;

    height: 50px !important;

    width: 100% !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    transition: 0.3s;
}

.stButton button:hover {

    transform: scale(1.02);
}

/* TICKER */

.ticker {
    width:100%;
    overflow:hidden;
    white-space:nowrap;
    box-sizing:border-box;
    color:#22c55e;
    font-size:18px;
    margin-bottom:15px;
}

.ticker span {
    display:inline-block;
    padding-left:100%;
    animation:ticker 18s linear infinite;
}

@keyframes ticker {
    0% {
        transform: translate3d(0,0,0);
    }
    100% {
        transform: translate3d(-100%,0,0);
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LIVE TICKER
# =========================================================
st.markdown("""
<div class='ticker'>
<span>
📈 AAPL +2.1% &nbsp;&nbsp;&nbsp;
🚀 TSLA +5.3% &nbsp;&nbsp;&nbsp;
💰 BTC +3.8% &nbsp;&nbsp;&nbsp;
📊 NVDA +4.2% &nbsp;&nbsp;&nbsp;
🔥 META +2.7%
</span>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("""
<div class='main-title'>
🌍 GLOBAL MARKETS AI
</div>

<div class='sub-title'>
Next Generation AI Stock Prediction Platform
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO IMAGE
# =========================================================
st.markdown("""
<div class='glass'>
    <img src='https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1600&auto=format&fit=crop'
    width='100%'
    style='border-radius:20px;'>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================================
# STOCK DICTIONARY
# =========================================================
stock_dict = {

    # US
    "apple": "AAPL",
    "tesla": "TSLA",
    "google": "GOOG",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",

    # INDIA
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",

    # CRYPTO
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD"
}

# =========================================================
# SEARCH SECTION
# =========================================================
st.markdown("<div class='search-box'>", unsafe_allow_html=True)

company_input = st.text_input(
    "🔍 Search Global Stocks",
    placeholder="Apple, Tesla, Google, Bitcoin..."
)

stock = ""

if company_input:

    search = company_input.strip().lower()

    if search in stock_dict:

        stock = stock_dict[search]

        st.success(f"✅ Symbol Detected: {stock}")

    else:

        st.warning("⚠ Company not found. Enter symbol manually.")

stock = st.text_input(
    "📌 Stock Symbol",
    value=stock,
    placeholder="AAPL, TSLA, GOOG"
)

search_btn = st.button("🚀 Predict Market")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MAIN PREDICTION
# =========================================================
if search_btn and stock != "":

    with st.spinner("🤖 AI analyzing market trends..."):

        # =================================================
        # MODEL
        # =================================================
        model = Sequential()

        model.add(LSTM(
            50,
            return_sequences=True,
            input_shape=(100,1)
        ))
        model.add(Dropout(0.2))

        model.add(LSTM(
            60,
            return_sequences=True
        ))
        model.add(Dropout(0.3))

        model.add(LSTM(
            80,
            return_sequences=True
        ))
        model.add(Dropout(0.4))

        model.add(LSTM(120))
        model.add(Dropout(0.5))

        model.add(Dense(1))

        model.load_weights("fixed_model.h5")

        # =================================================
        # DOWNLOAD DATA
        # =================================================
        start = '2012-01-01'
        end = pd.Timestamp.today()

        data = yf.download(stock, start, end)

        # =================================================
        # METRICS
        # =================================================
        latest_price = round(data.Close.iloc[-1], 2)
        highest_price = round(data.Close.max(), 2)
        lowest_price = round(data.Close.min(), 2)

        change = round(
            ((latest_price - lowest_price) / lowest_price) * 100,
            2
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💰 Latest</h3>
                <h1>${latest_price}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📈 Highest</h3>
                <h1>${highest_price}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📉 Lowest</h3>
                <h1>${lowest_price}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🚀 Growth</h3>
                <h1>{change}%</h1>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # =================================================
        # MOVING AVERAGES
        # =================================================
        ma50 = data.Close.rolling(50).mean()
        ma100 = data.Close.rolling(100).mean()

        st.subheader("📊 Market Trend Analysis")

        fig = plt.figure(figsize=(14,6))

        plt.plot(data.Close, label='Close Price')
        plt.plot(ma50, label='MA50')
        plt.plot(ma100, label='MA100')

        plt.grid(True)
        plt.legend()

        st.pyplot(fig)

        # =================================================
        # PREDICTION GRAPH
        # =================================================
        data_train = data.Close[
            0:int(len(data)*0.80)
        ]

        data_test = data.Close[
            int(len(data)*0.80):
        ]

        scaler = MinMaxScaler(
            feature_range=(0,1)
        )

        past_100_days = data_train.tail(100)

        data_test = pd.concat(
            [past_100_days, data_test],
            ignore_index=True
        )

        data_test_scale = scaler.fit_transform(
            data_test.values.reshape(-1,1)
        )

        x = []
        y = []

        for i in range(
            100,
            data_test_scale.shape[0]
        ):
            x.append(data_test_scale[i-100:i])
            y.append(data_test_scale[i,0])

        x = np.array(x)
        y = np.array(y)

        predict = model.predict(x)

        scale = 1 / scaler.scale_[0]

        predict = predict * scale
        y = y * scale

        st.subheader("🤖 AI Prediction Analysis")

        fig2 = plt.figure(figsize=(14,6))

        plt.plot(
            y,
            'green',
            label='Original Price'
        )

        plt.plot(
            predict,
            'red',
            label='Predicted Price'
        )

        plt.xlabel('Time')
        plt.ylabel('Price')

        plt.grid(True)
        plt.legend()

        st.pyplot(fig2)

        latest_prediction = round(
            float(predict[-1]),
            2
        )

        current_price = round(
            float(y[-1]),
            2
        )

        if latest_prediction > current_price:

            st.success("🟢 AI SIGNAL: BUY")

        else:

            st.error("🔴 AI SIGNAL: SELL")

        st.success(
            "✅ Market Prediction Completed"
        )