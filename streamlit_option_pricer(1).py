# streamlit_bs_pricer.py
# Simple Streamlit app for European option pricing with Black–Scholes formula.
# Uses Yahoo Finance data for real-world spot, strike, and expiry handling.
# Run: streamlit run streamlit_bs_pricer.py

import streamlit as st
from math import log, sqrt, exp, pi, erf
from datetime import date
import yfinance as yf

st.set_page_config(page_title="Black–Scholes Pricer", layout="wide")
st.title("🧮 Black–Scholes Option Pricer with Real Market Data")

st.markdown("""
This app prices European options (call/put) using the Black–Scholes formula.
You can either enter parameters manually or fetch real stock data via Yahoo Finance.
""")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Input method")
    mode = st.radio("Choose input method", ["Manual", "Yahoo Finance"])

    if mode == "Yahoo Finance":
        ticker = st.text_input("Ticker symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")
        expiry = st.date_input("Expiry date", value=date(2025, 12, 19))
        K = st.number_input("Strike price (K)", min_value=0.0, value=150.0, step=1.0)
        option_type = st.selectbox("Option type", ["Call", "Put"])

        st.markdown("---")
        st.caption("Spot price will be fetched live. r and q are assumed constant.")
    else:
        option_type = st.selectbox("Option type", ["Call", "Put"])
        S = st.number_input("Spot price (S)", min_value=0.0, value=100.0, step=1.0)
        K = st.number_input("Strike price (K)", min_value=0.0, value=100.0, step=1.0)
        days = st.number_input("Time to expiry (days)", min_value=0, value=30, step=1)
        expiry = date.today().toordinal() + days

    r_pct = st.number_input("Risk‑free rate (annual %, r)", value=1.0, step=0.1)
    q_pct = st.number_input("Dividend yield (annual %, q)", value=0.0, step=0.1)
    sigma_pct = st.number_input("Volatility (annual %, sigma)", value=20.0, step=0.1)

    r = r_pct / 100.0
    q = q_pct / 100.0
    sigma = sigma_pct / 100.0

# -------------------- Math helpers --------------------

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_pdf(x):
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

# -------------------- Black–Scholes pricing --------------------

def bs_price(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))

    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:
        return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)

# -------------------- Greeks --------------------

def bs_greeks(S, K, T, r, q, sigma, option_type="call"):
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    delta_call = exp(-q * T) * norm_cdf(d1)
    delta_put = exp(-q * T) * (norm_cdf(d1) - 1.0)
    delta = delta_call if option_type == "call" else delta_put

    gamma = exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * exp(-q * T) * norm_pdf(d1) * sqrt(T)

    theta_call = -(S * norm_pdf(d1) * sigma * exp(-q * T)) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm_cdf(d2) + q * S * exp(-q * T) * norm_cdf(d1)
    theta_put = -(S * norm_pdf(d1) * sigma * exp(-q * T)) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(-d2) - q * S * exp(-q * T) * norm_cdf(-d1)
    theta = theta_call if option_type == "call" else theta_put

    rho_call = K * T * exp(-r * T) * norm_cdf(d2)
    rho_put = -K * T * exp(-r * T) * norm_cdf(-d2)
    rho = rho_call if option_type == "call" else rho_put

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

# -------------------- Compute --------------------

if st.button("Price option"):
    if mode == "Yahoo Finance":
        data = yf.Ticker(ticker)
        spot = data.history(period="1d")["Close"].iloc[-1]
        S = float(spot)
        st.info(f"Fetched spot price for {ticker}: {S:.2f}")

        today = date.today()
        T = (expiry - today).days / 365.0
    else:
        today = date.today()
        T = (expiry - today.toordinal()) / 365.0

    ot = option_type.lower()
    price = bs_price(S, K, T, r, q, sigma, ot)
    greeks = bs_greeks(S, K, T, r, q, sigma, ot)

    st.success(f"Option price (Black–Scholes): {price:.4f}")
    st.write("Greeks:")
    st.json(greeks)
