import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm

# --- Black–Scholes helpers ---
def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
    )
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
    return delta, gamma, vega, theta, rho

# --- Streamlit Page ---
st.title("🛡️ Hedging Strategies Dashboard")

st.sidebar.header("Strategy Builder")

S = st.sidebar.number_input("Spot Price (S)", value=100.0, step=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.02, step=0.01)
T = st.sidebar.number_input("Time to Expiry (years)", value=0.5, step=0.1)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

st.sidebar.markdown("### Add Positions")

n_positions = st.sidebar.number_input("Number of Positions", min_value=1, max_value=5, value=2)

positions = []
for i in range(int(n_positions)):
    st.sidebar.markdown(f"**Position {i+1}**")
    option_type = st.sidebar.selectbox(f"Option Type {i+1}", ["call", "put"], key=f"type{i}")
    direction = st.sidebar.selectbox(f"Direction {i+1}", ["long", "short"], key=f"dir{i}")
    K = st.sidebar.number_input(f"Strike {i+1}", value=100.0, step=1.0, key=f"K{i}")
    qty = st.sidebar.number_input(f"Quantity {i+1}", value=1, step=1, key=f"qty{i}")
    positions.append((option_type, direction, K, qty))

# --- Compute Payoff ---
prices = np.linspace(0.5 * S, 1.5 * S, 200)
payoff = np.zeros_like(prices)
total_delta = total_gamma = total_vega = total_theta = total_rho = 0

for option_type, direction, K, qty in positions:
    sign = 1 if direction == "long" else -1
    # payoff at expiry
    if option_type == "call":
        payoff_leg = np.maximum(prices - K, 0)
    else:
        payoff_leg = np.maximum(K - prices, 0)
    payoff += sign * qty * payoff_leg
    
    # greeks now
    d, g, v, t, r_ = bs_greeks(S, K, T, r, sigma, option_type)
    total_delta += sign * qty * d
    total_gamma += sign * qty * g
    total_vega  += sign * qty * v
    total_theta += sign * qty * t
    total_rho   += sign * qty * r_

# --- Plot Payoff ---
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(prices, payoff, label="Strategy Payoff")
ax.axhline(0, color="black", linestyle="--")
ax.axvline(S, color="blue", linestyle=":", label="Spot Price")
ax.set_xlabel("Stock Price at Expiry")
ax.set_ylabel("Payoff")
ax.legend()
st.pyplot(fig)

# --- Display Combined Greeks ---
st.subheader("📊 Combined Greeks")
st.write(f"**Delta**: {total_delta:.3f}")
st.write(f"**Gamma**: {total_gamma:.3f}")
st.write(f"**Vega**: {total_vega:.3f}")
st.write(f"**Theta**: {total_theta:.3f}")
st.write(f"**Rho**: {total_rho:.3f}")
