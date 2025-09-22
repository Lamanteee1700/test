import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# --- Black-Scholes Functions ---
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

# --- PAGE: Option Pricer ---
st.title("📊 Black-Scholes Option Pricer with Greeks Dashboard")

ticker = st.text_input("Enter Stock Ticker (Yahoo Finance)", "AAPL")
data = yf.Ticker(ticker).history(period="1d")
S = data["Close"].iloc[-1]
st.write(f"Current {ticker} Price: **{S:.2f}**")

# --- Time, rates, vol ---
T_days = st.number_input("Time to Expiry (days)", value=30)
r = st.number_input("Risk-free Rate (e.g., 0.05 = 5%)", value=0.05)
sigma = st.number_input("Volatility (e.g., 0.2 = 20%)", value=0.2)
option_type = st.selectbox("Option Type", ["call", "put"])

# --- Strike price selection ---
st.subheader("Strike Price Selection")
strike_mode = st.radio(
    "Choose Strike Price Mode",
    ["ATM", "Deep ITM", "Deep OTM", "Custom"],
    index=0
)

if strike_mode == "ATM":
    K = round(S, 2)
elif strike_mode == "Deep ITM":
    K = round(S * (0.7 if option_type == "call" else 1.3), 2)
elif strike_mode == "Deep OTM":
    K = round(S * (1.3 if option_type == "call" else 0.7), 2)
else:
    K = st.number_input("Custom Strike Price", value=round(S, 2))

st.write(f"Selected Strike Price: **{K}**")
T = T_days/365

price = bs_price(S, K, T, r, sigma, option=option_type)
delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option=option_type)

st.subheader("💡 Option Price")
st.write(f"{option_type.capitalize()} Price: **{price:.2f}**")

# Greeks Dashboard
st.subheader("📊 Greeks Dashboard")
st.table({
    "Delta": [round(delta, 4)],
    "Gamma": [round(gamma, 4)],
    "Vega": [round(vega, 4)],
    "Theta": [round(theta, 4)],
    "Rho": [round(rho, 4)]
})

# Greeks Graph
st.subheader("📈 Greeks Sensitivity Graphs")
greek_choices = st.multiselect(
    "Choose Greeks to plot",
    ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    default=["Delta"]
)

S_range = np.linspace(S*0.7, S*1.3, 100)
greeks_dict = {g: [] for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}

for S_val in S_range:
    d, g, v, t, r_ = greeks(S_val, K, T, r, sigma, option=option_type)
    greeks_dict["Delta"].append(d)
    greeks_dict["Gamma"].append(g)
    greeks_dict["Vega"].append(v)
    greeks_dict["Theta"].append(t)
    greeks_dict["Rho"].append(r_)

fig, ax = plt.subplots()
for g in greek_choices:
    ax.plot(S_range, greeks_dict[g], label=g)
ax.axvline(K, color="red", linestyle="--", label="Strike Price")
ax.set_xlabel("Stock Price")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# Implied Volatility Calculator
st.subheader("📈 Implied Volatility Calculator")
market_price = st.number_input("Enter Market Option Price", min_value=0.0, step=0.1)
if market_price > 0:
    def option_price_given_vol(vol):
        return bs_price(S, K, T, r, vol, option_type) - market_price
    try:
        implied_vol = brentq(option_price_given_vol, 1e-6, 5.0)
        st.success(f"Implied Volatility: {implied_vol:.2%}")
    except ValueError:
        st.error("❌ Could not find implied volatility with given inputs.")
