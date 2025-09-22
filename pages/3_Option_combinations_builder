# --- PAGE 3: Option Combinations Builder ---

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import bs_price, greeks

# Page title
st.title("📐 Option Combinations Builder")

st.markdown("""
This tool lets you build and analyze common **option strategies**.  
Choose a strategy from the sidebar or create your own custom one.
""")

# Sidebar strategy selector
strategy = st.sidebar.selectbox(
    "Select Strategy",
    ["Custom", "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Iron Condor"]
)

# Parameters
st.subheader("Strategy Parameters")
S = st.number_input("Underlying Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T in years)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.01)
sigma = st.number_input("Volatility (σ)", value=0.2)

# Strategy logic
payoff = None
stock_prices = np.linspace(S * 0.5, S * 1.5, 200)

if strategy == "Custom":
    st.info("Define your own payoff by combining calls and puts manually (to be implemented).")

elif strategy == "Straddle":
    payoff = [max(sp - K, 0) + max(K - sp, 0) for sp in stock_prices]
    st.write("**Straddle:** Buy 1 Call and 1 Put at the same strike K.")

elif strategy == "Strangle":
    K_put = K * 0.95
    K_call = K * 1.05
    payoff = [max(sp - K_call, 0) + max(K_put - sp, 0) for sp in stock_prices]
    st.write("**Strangle:** Buy 1 OTM Call and 1 OTM Put.")

elif strategy == "Bull Call Spread":
    K_low = K
    K_high = K * 1.1
    payoff = [max(sp - K_low, 0) - max(sp - K_high, 0) for sp in stock_prices]
    st.write("**Bull Call Spread:** Buy Call at K, Sell Call at K_high.")

elif strategy == "Bear Put Spread":
    K_high = K
    K_low = K * 0.9
    payoff = [max(K_high - sp, 0) - max(K_low - sp, 0) for sp in stock_prices]
    st.write("**Bear Put Spread:** Buy Put at K_high, Sell Put at K_low.")

elif strategy == "Iron Condor":
    K1, K2, K3, K4 = K * 0.9, K * 0.95, K * 1.05, K * 1.1
    payoff = [
        -max(K1 - sp, 0) + max(K2 - sp, 0) + max(sp - K3, 0) - max(sp - K4, 0)
        for sp in stock_prices
    ]
    st.write("**Iron Condor:** Combination of Bull Put Spread and Bear Call Spread.")

# Plot
if payoff is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stock_prices, payoff, label=f"{strategy} Payoff", lw=2)
    ax.axvline(S, color="gray", linestyle="--", label="Spot Price")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Stock Price at Expiration")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(f"{strategy} Payoff Diagram")
    ax.legend()
    st.pyplot(fig)
