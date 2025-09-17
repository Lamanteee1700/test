# streamlit_bs_pricer.py
# Simple Streamlit app for European option pricing with Black–Scholes formula.
# Uses Yahoo Finance data for real-world spot, strike, and expiry handling.
# Run: streamlit run streamlit_bs_pricer.py

import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- Black-Scholes Greeks ---
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

def bs_price(S, K, T, r, sigma, option="call"):
    if option == "call":
        return S*norm.cdf(d1(S,K,T,r,sigma)) - K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)) - S*norm.cdf(-d1(S,K,T,r,sigma))

def greeks(S, K, T, r, sigma, option="call"):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    delta = norm.cdf(d1_val) if option=="call" else -norm.cdf(-d1_val)
    gamma = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1_val) * np.sqrt(T) / 100
    theta = (-(S*norm.pdf(d1_val)*sigma)/(2*np.sqrt(T)) 
             - r*K*np.exp(-r*T)*norm.cdf(d2_val if option=="call" else -d2_val)) / 365
    rho   = (K*T*np.exp(-r*T)*norm.cdf(d2_val if option=="call" else -d2_val)) / 100

    return delta, gamma, vega, theta, rho

# --- Streamlit App ---
st.title("📊 Black-Scholes Option Pricer with Greeks Dashboard")

ticker = st.text_input("Enter Stock Ticker (Yahoo Finance)", "AAPL")
data = yf.Ticker(ticker).history(period="1d")
S = data["Close"].iloc[-1]

st.write(f"Current {ticker} Price: **{S:.2f}**")

K = st.number_input("Strike Price", value=150.0)
T_days = st.number_input("Time to Expiry (days)", value=30)
r = st.number_input("Risk-free Rate (e.g., 0.05 = 5%)", value=0.05)
sigma = st.number_input("Volatility (e.g., 0.2 = 20%)", value=0.2)
option_type = st.selectbox("Option Type", ["call", "put"])

T = T_days/365

price = bs_price(S, K, T, r, sigma, option=option_type)
delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option=option_type)

st.subheader("💡 Option Price")
st.write(f"{option_type.capitalize()} Price: **{price:.2f}**")

# --- Greeks Dashboard ---
st.subheader("📊 Greeks Dashboard")
st.table({
    "Delta": [round(delta, 4)],
    "Gamma": [round(gamma, 4)],
    "Vega": [round(vega, 4)],
    "Theta": [round(theta, 4)],
    "Rho": [round(rho, 4)]
})

# --- Greeks Graph ---
st.subheader("📈 Greeks Sensitivity Graphs")
greek_choice = st.selectbox("Choose a Greek to plot", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

S_range = np.linspace(S*0.7, S*1.3, 100)
values = []

for S_val in S_range:
    d, g, v, t, r_ = greeks(S_val, K, T, r, sigma, option=option_type)
    if greek_choice == "Delta":
        values.append(d)
    elif greek_choice == "Gamma":
        values.append(g)
    elif greek_choice == "Vega":
        values.append(v)
    elif greek_choice == "Theta":
        values.append(t)
    elif greek_choice == "Rho":
        values.append(r_)

fig, ax = plt.subplots()
ax.plot(S_range, values, label=greek_choice)
ax.set_xlabel("Stock Price")
ax.set_ylabel(greek_choice)
ax.legend()
st.pyplot(fig)
