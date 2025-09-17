import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

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


# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Option Pricer", "Theory", "Option Combinations Builder", "Greeks Hedging Strategy", "Volatility Strategies"])

# --- PAGE 1: Option Pricer ---
if page == "Option Pricer":
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
        if option_type == "call":
            K = round(S * 0.7, 2)
        else:
            K = round(S * 1.3, 2)
    
    elif strike_mode == "Deep OTM":
        if option_type == "call":
            K = round(S * 1.3, 2)
        else:
            K = round(S * 0.7, 2)
    
    else:  # Custom
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
    greeks_dict = {"Delta": [], "Gamma": [], "Vega": [], "Theta": [], "Rho": []}

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


# --- PAGE 2: Theory ---
elif page == "Theory":
    st.title("📖 Black-Scholes Model: Theory & Assumptions")

    st.markdown("""
    ### Assumptions
    - The stock follows a **geometric Brownian motion** with constant drift and volatility.  
    - No arbitrage opportunities.  
    - Markets are frictionless (no transaction costs, infinite divisibility).  
    - Constant risk-free interest rate.  
    - European-style options (exercise only at expiry).  

    ### Black–Scholes PDE
    The option price \\( V(S,t) \\) satisfies:
    $$
    \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} 
    + rS \frac{\partial V}{\partial S} - rV = 0
    $$

    ### Closed-form Solution
    For a **European Call**:
    $$
    C(S, t) = S N(d_1) - K e^{-rT} N(d_2)
    $$

    For a **European Put**:
    $$
    P(S, t) = K e^{-rT} N(-d_2) - S N(-d_1)
    $$

    where:
    $$
    d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad
    d_2 = d_1 - \sigma \sqrt{T}
    $$

    ### Greeks
    - **Delta (Δ)**: sensitivity to stock price.  
    - **Gamma (Γ)**: rate of change of Delta.  
    - **Vega (ν)**: sensitivity to volatility.  
    - **Theta (Θ)**: sensitivity to time.  
    - **Rho (ρ)**: sensitivity to interest rate.
    
    ### Strike Price Scenarios  
    - **ATM (At the Money)** → strike ≈ current spot price.  
    - **Deep ITM (In the Money)**  
       - Call: strike ≈ 70% of spot (much cheaper than stock).  
       - Put: strike ≈ 130% of spot (much higher than stock).  
    - **Deep OTM (Out of the Money)**  
       - Call: strike ≈ 130% of spot (much more expensive than stock).  
       - Put: strike ≈ 70% of spot (much cheaper than stock).  
    """)

# --- PAGE 3: Option Combinations Builder ---
elif page == "Option Combinations Builder":
    st.title("🛡️ Option Combinations Builder")

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

    # Compute payoff and combined Greeks
    prices = np.linspace(0.5*S, 1.5*S, 200)
    payoff = np.zeros_like(prices)
    total_delta = total_gamma = total_vega = total_theta = total_rho = 0

    for option_type, direction, K, qty in positions:
        sign = 1 if direction=="long" else -1
        payoff_leg = np.maximum(prices - K,0) if option_type=="call" else np.maximum(K - prices,0)
        payoff += sign*qty*payoff_leg

        d,g,v,t,r_ = greeks(S,K,T,r,sigma,option_type)
        total_delta += sign*qty*d
        total_gamma += sign*qty*g
        total_vega  += sign*qty*v
        total_theta += sign*qty*t
        total_rho   += sign*qty*r_

    # Plot payoff
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(prices, payoff, label="Strategy Payoff")
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(S, color="blue", linestyle=":", label="Spot Price")
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Payoff")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Combined Greeks")
    st.write(f"**Delta**: {total_delta:.3f}")
    st.write(f"**Gamma**: {total_gamma:.3f}")
    st.write(f"**Vega**: {total_vega:.3f}")
    st.write(f"**Theta**: {total_theta:.3f}")
    st.write(f"**Rho**: {total_rho:.3f}")

# PAGE 4: Greeks Hedging Strategy
elif page == "Greeks Hedging Strategy":
    st.title("🧮 Greeks Hedging Strategies")
    
    st.markdown("""
    This page demonstrates **hedging Delta and Gamma step by step**:
    
    1. **Delta Hedging**: Buy/sell the underlying stock to neutralize Delta today.  
    2. **Gamma Hedging**: If Gamma ≠ 0, your Delta will drift as stock moves; we add a second option to reduce Gamma, stabilizing Delta over price changes.  
    """)

    st.subheader("Step 1: Select Your Option")
    S = st.number_input("Spot Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Time to Expiry (years)", value=0.5, step=0.1)
    r = st.number_input("Risk-Free Rate (r)", value=0.02, step=0.01)
    sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)
    option_type = st.selectbox("Option Type", ["call", "put"])

    # Original Greeks
    delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option_type)
    st.markdown(f"**Original Option Greeks:** Delta={delta:.3f}, Gamma={gamma:.3f}")

    # -------------------------------
    # DELTA HEDGING
    # -------------------------------
    st.subheader("Step 2: Delta Hedging")
    st.markdown("""
    Delta hedging involves buying/selling the underlying stock to neutralize the option's Delta **today**.
    """)
    delta_hedge_qty = -delta
    st.write(f"To delta-hedge, buy/sell **{delta_hedge_qty:.2f} shares** of the underlying.")

    # Visualize Delta hedged payoff
    S_range = np.linspace(S*0.7, S*1.3, 100)
    payoff_option = np.maximum(S_range - K,0) if option_type=="call" else np.maximum(K - S_range,0)
    payoff_delta_hedged = payoff_option + delta_hedge_qty*(S_range - S)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(S_range, payoff_option, label="Original Option")
    ax.plot(S_range, payoff_delta_hedged, label="Delta-Hedged Position")
    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(S, color="blue", linestyle=":", label="Spot Price")
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Payoff")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # GAMMA HEDGING
    # -------------------------------
    st.subheader("Step 3: Gamma Hedging")
    st.markdown("""
    Delta hedging works only at **today’s spot price**.  
    If Gamma ≠ 0, Delta will **drift** as the stock moves.  
    To stabilize Delta over a range of stock prices, we add a second option to reduce Gamma.  
    """)

    # Example: second option with different strike
    K2 = round(1.05*S,2)
    delta2, gamma2, _, _, _ = greeks(S, K2, T, r, sigma, option_type)
    gamma_hedge_qty = -gamma / gamma2 if gamma2 != 0 else 0
    st.write(f"Add **{gamma_hedge_qty:.2f} units of option with strike {K2}** to reduce Gamma to near zero.")

    # Visualize effect of Delta + Gamma hedge
    payoff_leg2 = (S_range - K2) if option_type=="call" else (K2 - S_range)
    payoff_gamma_hedged = payoff_option + delta_hedge_qty*(S_range - S) + gamma_hedge_qty*np.maximum(payoff_leg2, 0)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(S_range, payoff_option, label="Original Option")
    ax2.plot(S_range, payoff_delta_hedged, label="Delta-Hedged")
    ax2.plot(S_range, payoff_gamma_hedged, label="Delta + Gamma Hedged")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.axvline(S, color="blue", linestyle=":", label="Spot Price")
    ax2.set_xlabel("Stock Price at Expiry")
    ax2.set_ylabel("Payoff")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("""
    ✅ **Explanation:**  
    - The first hedge neutralizes Delta **today**, but as the stock moves, Delta drifts due to Gamma.  
    - Adding a second option reduces Gamma, keeping Delta more stable across price changes.  
    - This shows the principle of **dynamic hedging** in practice.
    """)

# Function to fetch real-time VIX data from Polygon.io
def fetch_vix_data(api_key):
    url = f'https://api.polygon.io/v2/aggs/prev/index/VIX?apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data['results'][0]['c'] if 'results' in data else None

# Function to calculate Black-Scholes option price
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- PAGE 5: Volatility Strategies ---
elif page == "Volatility Strategies":
    st.title("📈 Volatility Strategies: Vega, VIX, and Volatility Surfaces")
    
    st.markdown("""
    This page delves into volatility trading strategies, focusing on:
    
    - **Vega exposure**: Sensitivity to changes in implied volatility.
    - **VIX-based strategies**: Utilizing the Volatility Index to trade volatility.
    - **Volatility Smile**: Understanding how implied volatility varies with strike prices.
    - **3D Volatility Surface**: Visualizing implied volatility across different strikes and expirations.
    """)
    
    # Fetch and display real-time VIX data
    api_key = 'your_polygon_api_key'
    vix_value = fetch_vix_data(api_key)
    if vix_value:
        st.write(f"**Current VIX Index Value**: {vix_value}")
    else:
        st.write("Failed to retrieve real-time VIX data.")
    
    # Volatility Smile Visualization
    st.subheader("🔹 Volatility Smile")
    st.markdown("""
    The **Volatility Smile** refers to the pattern where implied volatility is higher for deep in-the-money (ITM) and out-of-the-money (OTM) options compared to at-the-money (ATM) options. This phenomenon is often observed in equity options and is attributed to factors like:
    
    - **Market sentiment**: Increased demand for protective puts during market uncertainty.
    - **Supply and demand dynamics**: Limited availability of certain strike prices can lead to higher premiums.
    
    Understanding the volatility smile is crucial for:
    
    - **Pricing options accurately**: Adjusting for varying implied volatilities across strikes.
    - **Identifying arbitrage opportunities**: Spotting mispricings between options with different strikes.
    """)
    
    # Generate and display Volatility Smile plot
    strikes = np.linspace(80, 120, 10)
    implied_vols = np.array([0.25, 0.28, 0.31, 0.33, 0.34, 0.33, 0.31, 0.28, 0.26, 0.25])
    fig_smile = go.Figure(data=[go.Scatter(x=strikes, y=implied_vols, mode='lines+markers', name='Implied Volatility')])
    fig_smile.update_layout(title="Volatility Smile", xaxis_title="Strike Price", yaxis_title="Implied Volatility")
    st.plotly_chart(fig_smile)
    
    # 3D Volatility Surface Visualization
    st.subheader("🔹 3D Volatility Surface")
    st.markdown("""
    A **3D Volatility Surface** is a graphical representation that shows how implied volatility changes across different strike prices and expiration dates. It provides a comprehensive view of market expectations and can be used to:
    
    - **Analyze term structure**: Understand how volatility expectations change over time.
    - **Identify skewness**: Detect how implied volatility varies with strike prices.
    - **Assess market sentiment**: Gauge investor expectations for future volatility.
    """)
    
    # Generate and display 3D Volatility Surface plot
    strikes_3d = np.linspace(80, 120, 10)
    maturities = np.linspace(30, 365, 10)
    X, Y = np.meshgrid(strikes_3d, maturities)
    Z = np.sin(X / 10) * np.cos(Y / 100) + 0.2  # Example volatility surface function
    
    fig_surface = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig_surface.update_layout(title="3D Volatility Surface", scene=dict(
        xaxis_title='Strike Price',
        yaxis_title='Time to Expiration (Days)',
        zaxis_title='Implied Volatility'
    ))
    st.plotly_chart(fig_surface)
    
    # Vega Exposure Simulation
    st.subheader("🔹 Vega Exposure")
    st.markdown("""
    **Vega** measures how the option price changes with a 1% change in implied volatility. Understanding vega exposure is crucial for:
    
    - **Hedging volatility risk**: Adjusting positions to mitigate the impact of volatility changes.
    - **Speculating on volatility**: Taking positions based on expectations of volatility movements.
    
    Let's simulate how option price changes with volatility:
    """)
    
    # Option parameters
    S = st.number_input("Spot Price (S)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Time to Expiry (years)", value=0.5, step=0.1)
    r = st.number_input("Risk-Free Rate (r)", value=0.02, step=0.01)
    option_type = st.selectbox("Option Type", ["call", "put"])
    sigma_base = st.number_input("Base Volatility (σ)", value=0.2, step=0.01)
    
    # Calculate option price for different volatilities
    sigma_range = np.linspace(0.1, 0.5, 50)
    price_range = [black_scholes(S, K, T, r, s, option_type) for s in sigma_range]
    
    # Plot the Vega exposure
    fig_vega = go.Figure(data=[go.Scatter(x=sigma_range, y=price_range, mode='lines', name='Option Price')])
    fig_vega.update_layout(title="Option Price vs Implied Volatility", xaxis_title="Implied Volatility", yaxis_title="Option Price")
    st.plotly_chart(fig_vega)

    st.markdown("""
    As shown, the option price increases with volatility, indicating positive vega exposure.
    """)
