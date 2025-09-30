import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

def exotic_options_page():
    st.title("🎨 Exotic Options Pricing & Analytics")
    
    # Banner
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin-bottom: 20px;">
    <strong>Exotic Options:</strong> Price and analyze path-dependent and complex derivative instruments including 
    Asian, Barrier, Lookback, Binary, and Chooser options using Monte Carlo simulation and analytical methods.
    </div>
    """, unsafe_allow_html=True)
    
    # === MARKET DATA SECTION ===
    st.subheader("📊 Market Parameters")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio("Data Source", ["Live Data (Ticker)", "Custom Parameters"], horizontal=True)
        
        if data_source == "Live Data (Ticker)":
            ticker = st.text_input("Ticker Symbol", "AAPL", help="Enter stock ticker")
            
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="60d")
                
                if not data.empty:
                    S = data["Close"].iloc[-1]
                    
                    # Calculate historical volatility
                    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                    sigma = returns.std() * np.sqrt(252)
                    
                    st.success(f"✅ {ticker}: ${S:.2f} | HV: {sigma:.1%}")
                else:
                    st.warning("Using default values")
                    S = 100.0
                    sigma = 0.25
            except:
                st.warning("Error fetching data, using defaults")
                S = 100.0
                sigma = 0.25
        else:
            S = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, step=1.0)
            sigma = st.slider("Volatility (%)", 5.0, 100.0, 25.0, 2.5) / 100
    
    with col2:
        r = st.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.25) / 100
        q = st.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, 0.25) / 100
        
        st.metric("Spot Price", f"${S:.2f}")
        st.metric("Volatility", f"{sigma:.1%}")
    
    # Common parameters
    st.markdown("---")
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        T = st.slider("Time to Maturity (Years)", 0.1, 3.0, 1.0, 0.1)
        K = st.number_input("Strike Price ($)", value=float(S), min_value=0.01, step=1.0)
    
    with param_col2:
        option_type = st.selectbox("Option Type", ["call", "put"])
        n_simulations = st.select_slider("Monte Carlo Simulations", 
                                        options=[1000, 5000, 10000, 50000, 100000],
                                        value=10000,
                                        help="More simulations = more accurate but slower")
    
    st.markdown("---")
    
    # === EXOTIC OPTION TYPE SELECTION ===
    st.subheader("🎯 Select Exotic Option Type")
    
    exotic_type = st.selectbox(
        "Exotic Option",
        ["Asian Options", "Barrier Options", "Lookback Options", "Binary/Digital Options", 
         "Chooser Options", "Rainbow Options (2 Assets)"],
        help="Choose the type of exotic option to price"
    )
    
    st.markdown("---")
    
    # ========================================
    # ASIAN OPTIONS
    # ========================================
    if exotic_type == "Asian Options":
        st.markdown("### 🌏 Asian Options")
        st.info("**Asian options** have payoffs that depend on the average price of the underlying over a period. Less volatile than standard options.")
        
        asian_col1, asian_col2 = st.columns(2)
        
        with asian_col1:
            asian_type = st.radio("Averaging Type", ["Arithmetic", "Geometric"], horizontal=True)
            averaging_period = st.slider("Averaging Observations", 10, 252, 50, 
                                       help="Number of price observations for averaging")
        
        with asian_col2:
            asian_style = st.radio("Style", ["Average Price", "Average Strike"], horizontal=True,
                                  help="Average Price: payoff based on avg vs strike. Average Strike: payoff based on final vs avg")
        
        # Monte Carlo pricing for Asian options
        with st.spinner("Calculating Asian option price..."):
            np.random.seed(42)
            dt = T / averaging_period
            
            # Generate price paths
            paths = np.zeros((n_simulations, averaging_period + 1))
            paths[:, 0] = S
            
            for t in range(1, averaging_period + 1):
                z = np.random.standard_normal(n_simulations)
                paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
            # Calculate average
            if asian_type == "Arithmetic":
                avg_price = np.mean(paths[:, 1:], axis=1)
            else:  # Geometric
                avg_price = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            
            # Calculate payoff
            if asian_style == "Average Price":
                if option_type == "call":
                    payoffs = np.maximum(avg_price - K, 0)
                else:
                    payoffs = np.maximum(K - avg_price, 0)
            else:  # Average Strike
                if option_type == "call":
                    payoffs = np.maximum(paths[:, -1] - avg_price, 0)
                else:
                    payoffs = np.maximum(avg_price - paths[:, -1], 0)
            
            # Discount to present value
            asian_price = np.exp(-r * T) * np.mean(payoffs)
            asian_std = np.std(payoffs) * np.exp(-r * T)
            asian_se = asian_std / np.sqrt(n_simulations)
            
            # Display results
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Option Price", f"${asian_price:.4f}")
            result_col2.metric("Std Error", f"${asian_se:.4f}")
            result_col3.metric("95% CI", f"±${1.96*asian_se:.4f}")
            result_col4.metric("Avg Price", f"${np.mean(avg_price):.2f}")
            
            # Visualization
            viz_tab1, viz_tab2 = st.tabs(["Sample Paths", "Payoff Distribution"])
            
            with viz_tab1:
                fig_paths = go.Figure()
                
                # Plot sample paths
                num_paths_display = min(100, n_simulations)
                time_steps = np.linspace(0, T, averaging_period + 1)
                
                for i in range(num_paths_display):
                    fig_paths.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths[i],
                        mode='lines',
                        line=dict(width=1),
                        opacity=0.3,
                        showlegend=False
                    ))
                
                fig_paths.add_hline(y=K, line_dash="dash", line_color="red",
                                   annotation_text="Strike")
                fig_paths.add_hline(y=S, line_dash="dot", line_color="green",
                                   annotation_text="Spot")
                
                fig_paths.update_layout(
                    title="Sample Price Paths for Asian Option",
                    xaxis_title="Time (Years)",
                    yaxis_title="Price ($)",
                    height=400
                )
                st.plotly_chart(fig_paths, use_container_width=True)
            
            with viz_tab2:
                fig_dist = go.Figure()
                
                fig_dist.add_trace(go.Histogram(
                    x=payoffs,
                    nbinsx=50,
                    name="Payoff Distribution",
                    marker_color='blue'
                ))
                
                fig_dist.add_vline(x=asian_price * np.exp(r*T), line_dash="dash", 
                                  line_color="red",
                                  annotation_text=f"Mean: ${asian_price * np.exp(r*T):.2f}")
                
                fig_dist.update_layout(
                    title="Payoff Distribution at Maturity",
                    xaxis_title="Payoff ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # ========================================
    # BARRIER OPTIONS
    # ========================================
    elif exotic_type == "Barrier Options":
        st.markdown("### 🚧 Barrier Options")
        st.info("**Barrier options** activate (knock-in) or deactivate (knock-out) when the underlying crosses a barrier level.")
        
        barrier_col1, barrier_col2 = st.columns(2)
        
        with barrier_col1:
            barrier_direction = st.radio("Barrier Type", ["Up", "Down"], horizontal=True)
            barrier_style = st.radio("Barrier Style", ["Knock-Out", "Knock-In"], horizontal=True)
        
        with barrier_col2:
            if barrier_direction == "Up":
                H = st.number_input("Barrier Level ($)", value=float(S*1.2), min_value=float(S))
            else:
                H = st.number_input("Barrier Level ($)", value=float(S*0.8), max_value=float(S))
            
            rebate = st.number_input("Rebate ($)", value=0.0, min_value=0.0, 
                                    help="Payment if barrier is hit (knock-out) or not hit (knock-in)")
        
        # Monte Carlo simulation
        with st.spinner("Simulating barrier option..."):
            np.random.seed(42)
            n_steps = 252
            dt = T / n_steps
            
            paths = np.zeros((n_simulations, n_steps + 1))
            paths[:, 0] = S
            
            for t in range(1, n_steps + 1):
                z = np.random.standard_normal(n_simulations)
                paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
            # Check barrier conditions
            if barrier_direction == "Up":
                barrier_hit = np.max(paths, axis=1) >= H
            else:
                barrier_hit = np.min(paths, axis=1) <= H
            
            # Calculate payoffs
            final_prices = paths[:, -1]
            if option_type == "call":
                intrinsic = np.maximum(final_prices - K, 0)
            else:
                intrinsic = np.maximum(K - final_prices, 0)
            
            if barrier_style == "Knock-Out":
                # Option is worthless if barrier is hit
                payoffs = np.where(barrier_hit, rebate, intrinsic)
            else:  # Knock-In
                # Option only has value if barrier is hit
                payoffs = np.where(barrier_hit, intrinsic, rebate)
            
            barrier_price = np.exp(-r * T) * np.mean(payoffs)
            barrier_std = np.std(payoffs) * np.exp(-r * T)
            barrier_se = barrier_std / np.sqrt(n_simulations)
            
            prob_barrier_hit = np.mean(barrier_hit)
            
            # Results
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Option Price", f"${barrier_price:.4f}")
            result_col2.metric("Std Error", f"${barrier_se:.4f}")
            result_col3.metric("Barrier Hit Prob", f"{prob_barrier_hit:.2%}")
            result_col4.metric("Avg Payoff", f"${np.mean(payoffs):.2f}")
            
            # Visualization
            viz_tab1, viz_tab2 = st.tabs(["Price Paths", "Barrier Analysis"])
            
            with viz_tab1:
                fig_barrier = go.Figure()
                
                # Plot sample paths - color by barrier hit
                num_display = min(50, n_simulations)
                time_steps = np.linspace(0, T, n_steps + 1)
                
                for i in range(num_display):
                    color = 'red' if barrier_hit[i] else 'green'
                    opacity = 0.6 if barrier_hit[i] else 0.3
                    
                    fig_barrier.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths[i],
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=opacity,
                        showlegend=False
                    ))
                
                # Add barrier line
                fig_barrier.add_hline(y=H, line_dash="solid", line_color="black", line_width=3,
                                     annotation_text=f"Barrier: ${H:.2f}")
                fig_barrier.add_hline(y=K, line_dash="dash", line_color="blue",
                                     annotation_text=f"Strike: ${K:.2f}")
                
                fig_barrier.update_layout(
                    title=f"{barrier_direction}-and-{barrier_style} Barrier Option Paths",
                    xaxis_title="Time (Years)",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig_barrier, use_container_width=True)
                
                st.caption("🔴 Red paths: Barrier hit | 🟢 Green paths: Barrier not hit")
            
            with viz_tab2:
                # Distribution comparison
                fig_dist = make_subplots(rows=1, cols=2,
                                        subplot_titles=("Payoff Distribution", "Price at Maturity"))
                
                fig_dist.add_trace(go.Histogram(x=payoffs, nbinsx=50, name="Payoff",
                                               marker_color='blue'), row=1, col=1)
                
                fig_dist.add_trace(go.Histogram(x=final_prices, nbinsx=50, name="Final Price",
                                               marker_color='green'), row=1, col=2)
                
                fig_dist.update_xaxes(title_text="Payoff ($)", row=1, col=1)
                fig_dist.update_xaxes(title_text="Price ($)", row=1, col=2)
                fig_dist.update_yaxes(title_text="Frequency", row=1, col=1)
                
                fig_dist.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # ========================================
    # LOOKBACK OPTIONS
    # ========================================
    elif exotic_type == "Lookback Options":
        st.markdown("### 👀 Lookback Options")
        st.info("**Lookback options** have payoffs based on the maximum or minimum price reached during the option's life. Eliminates timing risk.")
        
        lookback_col1, lookback_col2 = st.columns(2)
        
        with lookback_col1:
            lookback_type = st.radio("Lookback Type", ["Fixed Strike", "Floating Strike"], horizontal=True)
        
        with lookback_col2:
            if lookback_type == "Fixed Strike":
                lookback_feature = st.radio("Feature", ["Max Price", "Min Price"], horizontal=True)
        
        # Monte Carlo simulation
        with st.spinner("Simulating lookback option..."):
            np.random.seed(42)
            n_steps = 252
            dt = T / n_steps
            
            paths = np.zeros((n_simulations, n_steps + 1))
            paths[:, 0] = S
            
            for t in range(1, n_steps + 1):
                z = np.random.standard_normal(n_simulations)
                paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
            max_prices = np.max(paths, axis=1)
            min_prices = np.min(paths, axis=1)
            final_prices = paths[:, -1]
            
            # Calculate payoffs
            if lookback_type == "Fixed Strike":
                if lookback_feature == "Max Price":
                    # Call on maximum: max(max_price - K, 0)
                    payoffs = np.maximum(max_prices - K, 0)
                else:
                    # Put on minimum: max(K - min_price, 0)
                    payoffs = np.maximum(K - min_prices, 0)
            else:  # Floating Strike
                if option_type == "call":
                    # Payoff: max(S_T - min_price, 0)
                    payoffs = np.maximum(final_prices - min_prices, 0)
                else:
                    # Payoff: max(max_price - S_T, 0)
                    payoffs = np.maximum(max_prices - final_prices, 0)
            
            lookback_price = np.exp(-r * T) * np.mean(payoffs)
            lookback_std = np.std(payoffs) * np.exp(-r * T)
            lookback_se = lookback_std / np.sqrt(n_simulations)
            
            # Results
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Option Price", f"${lookback_price:.4f}")
            result_col2.metric("Std Error", f"${lookback_se:.4f}")
            result_col3.metric("Avg Max Price", f"${np.mean(max_prices):.2f}")
            result_col4.metric("Avg Min Price", f"${np.mean(min_prices):.2f}")
            
            # Visualization
            viz_tab1, viz_tab2 = st.tabs(["Sample Paths with Extremes", "Payoff Analysis"])
            
            with viz_tab1:
                fig_lookback = go.Figure()
                
                num_display = min(20, n_simulations)
                time_steps = np.linspace(0, T, n_steps + 1)
                
                for i in range(num_display):
                    fig_lookback.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths[i],
                        mode='lines',
                        line=dict(width=1.5, color='lightblue'),
                        opacity=0.5,
                        showlegend=False
                    ))
                    
                    # Mark max and min points
                    max_idx = np.argmax(paths[i])
                    min_idx = np.argmin(paths[i])
                    
                    fig_lookback.add_trace(go.Scatter(
                        x=[time_steps[max_idx]],
                        y=[paths[i, max_idx]],
                        mode='markers',
                        marker=dict(size=8, color='green', symbol='star'),
                        showlegend=False
                    ))
                    
                    fig_lookback.add_trace(go.Scatter(
                        x=[time_steps[min_idx]],
                        y=[paths[i, min_idx]],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='star'),
                        showlegend=False
                    ))
                
                if lookback_type == "Fixed Strike":
                    fig_lookback.add_hline(y=K, line_dash="dash", line_color="black",
                                          annotation_text=f"Strike: ${K:.2f}")
                
                fig_lookback.update_layout(
                    title="Lookback Option: Price Paths with Max/Min Points",
                    xaxis_title="Time (Years)",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig_lookback, use_container_width=True)
                
                st.caption("⭐ Green stars: Maximum price | ⭐ Red stars: Minimum price")
            
            with viz_tab2:
                fig_payoff = make_subplots(rows=1, cols=2,
                                          subplot_titles=("Payoff Distribution", "Extreme Prices"))
                
                fig_payoff.add_trace(go.Histogram(x=payoffs, nbinsx=50, 
                                                 marker_color='purple'), row=1, col=1)
                
                fig_payoff.add_trace(go.Box(y=max_prices, name="Max", marker_color='green'), 
                                    row=1, col=2)
                fig_payoff.add_trace(go.Box(y=min_prices, name="Min", marker_color='red'), 
                                    row=1, col=2)
                
                fig_payoff.update_xaxes(title_text="Payoff ($)", row=1, col=1)
                fig_payoff.update_yaxes(title_text="Frequency", row=1, col=1)
                fig_payoff.update_yaxes(title_text="Price ($)", row=1, col=2)
                
                fig_payoff.update_layout(height=400)
                st.plotly_chart(fig_payoff, use_container_width=True)
    
    # ========================================
    # BINARY/DIGITAL OPTIONS
    # ========================================
    elif exotic_type == "Binary/Digital Options":
        st.markdown("### 💎 Binary/Digital Options")
        st.info("**Binary options** pay a fixed amount if a condition is met, otherwise nothing. All-or-nothing payoff structure.")
        
        binary_col1, binary_col2 = st.columns(2)
        
        with binary_col1:
            binary_type = st.radio("Binary Type", ["Cash-or-Nothing", "Asset-or-Nothing"], horizontal=True)
            payout_amount = st.number_input("Fixed Payout ($)", value=100.0, min_value=0.01) if binary_type == "Cash-or-Nothing" else None
        
        with binary_col2:
            st.metric("Strike Price", f"${K:.2f}")
            st.metric("Option Type", option_type.title())
        
        # Analytical pricing for binary options
        with st.spinner("Calculating binary option price..."):
            d1_val = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2_val = d1_val - sigma*np.sqrt(T)
            
            if binary_type == "Cash-or-Nothing":
                if option_type == "call":
                    binary_price = payout_amount * np.exp(-r*T) * norm.cdf(d2_val)
                    prob_itm = norm.cdf(d2_val)
                else:
                    binary_price = payout_amount * np.exp(-r*T) * norm.cdf(-d2_val)
                    prob_itm = norm.cdf(-d2_val)
            else:  # Asset-or-Nothing
                if option_type == "call":
                    binary_price = S * np.exp(-q*T) * norm.cdf(d1_val)
                    prob_itm = norm.cdf(d2_val)
                else:
                    binary_price = S * np.exp(-q*T) * norm.cdf(-d1_val)
                    prob_itm = norm.cdf(-d2_val)
            
            # Results
            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Option Price", f"${binary_price:.4f}")
            result_col2.metric("Prob ITM", f"{prob_itm:.2%}")
            
            if binary_type == "Cash-or-Nothing":
                expected_payout = payout_amount * prob_itm
                result_col3.metric("Expected Payout", f"${expected_payout:.2f}")
            else:
                expected_payout = S * prob_itm
                result_col3.metric("Expected Asset Value", f"${expected_payout:.2f}")
            
            # Visualization
            viz_tab1, viz_tab2 = st.tabs(["Payoff Diagram", "Price Sensitivity"])
            
            with viz_tab1:
                fig_binary_payoff = go.Figure()
                
                S_range = np.linspace(S*0.5, S*1.5, 200)
                
                if binary_type == "Cash-or-Nothing":
                    if option_type == "call":
                        payoffs_plot = np.where(S_range >= K, payout_amount, 0)
                    else:
                        payoffs_plot = np.where(S_range <= K, payout_amount, 0)
                else:
                    if option_type == "call":
                        payoffs_plot = np.where(S_range >= K, S_range, 0)
                    else:
                        payoffs_plot = np.where(S_range <= K, S_range, 0)
                
                fig_binary_payoff.add_trace(go.Scatter(
                    x=S_range,
                    y=payoffs_plot,
                    mode='lines',
                    name='Payoff at Maturity',
                    line=dict(color='blue', width=3)
                ))
                
                fig_binary_payoff.add_vline(x=K, line_dash="dash", line_color="red",
                                           annotation_text=f"Strike: ${K:.2f}")
                fig_binary_payoff.add_vline(x=S, line_dash="dot", line_color="green",
                                           annotation_text=f"Current: ${S:.2f}")
                
                fig_binary_payoff.update_layout(
                    title=f"Binary {option_type.title()} Payoff Diagram",
                    xaxis_title="Stock Price at Maturity ($)",
                    yaxis_title="Payoff ($)",
                    height=400
                )
                st.plotly_chart(fig_binary_payoff, use_container_width=True)
            
            with viz_tab2:
                # Price as function of spot
                prices_vs_spot = []
                for S_val in S_range:
                    d1_temp = (np.log(S_val/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                    d2_temp = d1_temp - sigma*np.sqrt(T)
                    
                    if binary_type == "Cash-or-Nothing":
                        if option_type == "call":
                            price_temp = payout_amount * np.exp(-r*T) * norm.cdf(d2_temp)
                        else:
                            price_temp = payout_amount * np.exp(-r*T) * norm.cdf(-d2_temp)
                    else:
                        if option_type == "call":
                            price_temp = S_val * np.exp(-q*T) * norm.cdf(d1_temp)
                        else:
                            price_temp = S_val * np.exp(-q*T) * norm.cdf(-d1_temp)
                    
                    prices_vs_spot.append(price_temp)
                
                fig_sensitivity = go.Figure()
                
                fig_sensitivity.add_trace(go.Scatter(
                    x=S_range,
                    y=prices_vs_spot,
                    mode='lines',
                    name='Option Price',
                    line=dict(color='purple', width=3)
                ))
                
                fig_sensitivity.add_vline(x=K, line_dash="dash", line_color="red")
                fig_sensitivity.add_vline(x=S, line_dash="dot", line_color="green")
                
                fig_sensitivity.update_layout(
                    title="Option Price vs Spot Price",
                    xaxis_title="Spot Price ($)",
                    yaxis_title="Option Value ($)",
                    height=400
                )
                st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    # ========================================
    # CHOOSER OPTIONS
    # ========================================
    elif exotic_type == "Chooser Options":
        st.markdown("### 🔀 Chooser Options")
        st.info("**Chooser options** allow the holder to choose whether the option is a call or put at a future date. Flexible for uncertain market conditions.")
        
        chooser_col1, chooser_col2 = st.columns(2)
        
        with chooser_col1:
            T_choose = st.slider("Time to Choose (Years)", 0.1, float(T-0.1), float(T/2), 0.1,
                               help="When to decide between call or put")
        
        with chooser_col2:
            st.metric("Maturity", f"{T:.2f} years")
            st.metric("Strike", f"${K:.2f}")
        
        # Analytical pricing for simple chooser
        with st.spinner("Calculating chooser option..."):
            # At choice time, holder will choose call if S > K, put otherwise
            # Chooser = Call(S, K, T) + Put(S, K, T_choose)
            
            d1_full = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2_full = d1_full - sigma*np.sqrt(T)
            
            d1_choose = (np.log(S/K) + (r - q + 0.5*sigma**2)*T_choose) / (sigma*np.sqrt(T_choose))
            d2_choose = d1_choose - sigma*np.sqrt(T_choose)
            
            # Simple chooser: max(Call, Put) at T_choose
            # Price = Call(S,K,T) + Put(S,K,T_choose) using put-call parity adjustment
            call_price = S*np.exp(-q*T)*norm.cdf(d1_full) - K*np.exp(-r*T)*norm.cdf(d2_full)
            put_price_choose = K*np.exp(-r*T_choose)*norm.cdf(-d2_choose) - S*np.exp(-q*T_choose)*norm.cdf(-d1_choose)
            
            chooser_price = call_price + put_price_choose
            
            # Monte Carlo for verification
            np.random.seed(42)
            n_steps_choose = int(T_choose * 252)
            n_steps_total = int(T * 252)
            dt = T / n_steps_total
            
            paths = np.zeros((n_simulations, n_steps_total + 1))
            paths[:, 0] = S
            
            for t in range(1, n_steps_total + 1):
                z = np.random.standard_normal(n_simulations)
                paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            
            # Price at choice time
            S_choose = paths[:, n_steps_choose]
            S_maturity = paths[:, -1]
            
            # At choice time, calculate value of call and put
            T_remaining = T - T_choose
            call_values = np.maximum(S_maturity - K, 0)
            put_values = np.maximum(K - S_maturity, 0)
            
            # Choose the better option
            chosen_payoffs = np.maximum(call_values, put_values)
            chooser_price_mc = np.exp(-r * T) * np.mean(chosen_payoffs)
            chooser_se = np.std(chosen_payoffs) * np.exp(-r * T) / np.sqrt(n_simulations)
            
            # Results
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Analytical Price", f"${chooser_price:.4f}")
            result_col2.metric("Monte Carlo Price", f"${chooser_price_mc:.4f}")
            result_col3.metric("Std Error", f"${chooser_se:.4f}")
            
            prob_choose_call = np.mean(call_values > put_values)
            result_col4.metric("Prob Choose Call", f"{prob_choose_call:.1%}")
            
            # Visualization
            viz_tab1, viz_tab2 = st.tabs(["Choice Analysis", "Price Paths"])
            
            with viz_tab1:
                fig_choice = make_subplots(rows=1, cols=2,
                                          subplot_titles=("Call vs Put Values", "Optimal Choice"))
                
                # Scatter plot of call vs put values
                fig_choice.add_trace(go.Scatter(
                    x=call_values[:500],
                    y=put_values[:500],
                    mode='markers',
                    marker=dict(size=4, color=call_values[:500] > put_values[:500],
                              colorscale=['red', 'green'], showscale=False),
                    showlegend=False
                ), row=1, col=1)
                
                # Add diagonal line
                max_val = max(np.max(call_values[:500]), np.max(put_values[:500]))
                fig_choice.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ), row=1, col=1)
                
                # Distribution of chosen payoffs
                fig_choice.add_trace(go.Histogram(
                    x=chosen_payoffs,
                    nbinsx=50,
                    marker_color='purple',
                    showlegend=False
                ), row=1, col=2)
                
                fig_choice.update_xaxes(title_text="Call Value ($)", row=1, col=1)
                fig_choice.update_yaxes(title_text="Put Value ($)", row=1, col=1)
                fig_choice.update_xaxes(title_text="Chosen Payoff ($)", row=1, col=2)
                fig_choice.update_yaxes(title_text="Frequency", row=1, col=2)
                
                fig_choice.update_layout(height=400)
                st.plotly_chart(fig_choice, use_container_width=True)
                
                st.caption("🟢 Green: Call chosen | 🔴 Red: Put chosen")
            
            with viz_tab2:
                fig_paths = go.Figure()
                
                num_display = min(50, n_simulations)
                time_steps = np.linspace(0, T, n_steps_total + 1)
                
                for i in range(num_display):
                    color = 'green' if call_values[i] > put_values[i] else 'red'
                    
                    fig_paths.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths[i],
                        mode='lines',
                        line=dict(color=color, width=1.5),
                        opacity=0.4,
                        showlegend=False
                    ))
                
                fig_paths.add_vline(x=T_choose, line_dash="solid", line_color="black", line_width=2,
                                   annotation_text=f"Choice Time: {T_choose:.2f}y")
                fig_paths.add_hline(y=K, line_dash="dash", line_color="blue",
                                   annotation_text=f"Strike: ${K:.2f}")
                
                fig_paths.update_layout(
                    title="Price Paths: Green=Call Chosen, Red=Put Chosen",
                    xaxis_title="Time (Years)",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig_paths, use_container_width=True)
    
    # ========================================
    # RAINBOW OPTIONS (2 ASSETS)
    # ========================================
    elif exotic_type == "Rainbow Options (2 Assets)":
        st.markdown("### 🌈 Rainbow Options (2 Assets)")
        st.info("**Rainbow options** have payoffs depending on multiple underlying assets. Common types include best-of, worst-of, and spread options.")
        
        rainbow_col1, rainbow_col2 = st.columns(2)
        
        with rainbow_col1:
            st.markdown("**Asset 1 Parameters**")
            S2 = st.number_input("Asset 2 Price ($)", value=float(S), min_value=0.01, step=1.0)
            sigma2 = st.slider("Asset 2 Volatility (%)", 5.0, 100.0, 25.0, 2.5, key='sigma2') / 100
        
        with rainbow_col2:
            st.markdown("**Correlation & Type**")
            correlation = st.slider("Correlation", -0.99, 0.99, 0.5, 0.05,
                                   help="Correlation between the two assets")
            rainbow_type = st.selectbox("Rainbow Type", 
                                        ["Best-of (Max)", "Worst-of (Min)", "Spread Option"])
        
        # Monte Carlo simulation
        with st.spinner("Simulating rainbow option..."):
            np.random.seed(42)
            n_steps = 252
            dt = T / n_steps
            
            # Generate correlated random numbers
            paths1 = np.zeros((n_simulations, n_steps + 1))
            paths2 = np.zeros((n_simulations, n_steps + 1))
            paths1[:, 0] = S
            paths2[:, 0] = S2
            
            # Cholesky decomposition for correlation
            chol = np.linalg.cholesky([[1, correlation], [correlation, 1]])
            
            for t in range(1, n_steps + 1):
                z = np.random.standard_normal((n_simulations, 2))
                z_corr = z @ chol.T
                
                paths1[:, t] = paths1[:, t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z_corr[:, 0])
                paths2[:, t] = paths2[:, t-1] * np.exp((r - q - 0.5*sigma2**2)*dt + sigma2*np.sqrt(dt)*z_corr[:, 1])
            
            final_prices1 = paths1[:, -1]
            final_prices2 = paths2[:, -1]
            
            # Calculate payoffs based on rainbow type
            if rainbow_type == "Best-of (Max)":
                if option_type == "call":
                    payoffs = np.maximum(np.maximum(final_prices1, final_prices2) - K, 0)
                else:
                    payoffs = np.maximum(K - np.maximum(final_prices1, final_prices2), 0)
                
            elif rainbow_type == "Worst-of (Min)":
                if option_type == "call":
                    payoffs = np.maximum(np.minimum(final_prices1, final_prices2) - K, 0)
                else:
                    payoffs = np.maximum(K - np.minimum(final_prices1, final_prices2), 0)
                
            else:  # Spread Option
                # Payoff = max(S1 - S2 - K, 0) for call, max(K - (S1 - S2), 0) for put
                spread = final_prices1 - final_prices2
                if option_type == "call":
                    payoffs = np.maximum(spread - K, 0)
                else:
                    payoffs = np.maximum(K - spread, 0)
            
            rainbow_price = np.exp(-r * T) * np.mean(payoffs)
            rainbow_std = np.std(payoffs) * np.exp(-r * T)
            rainbow_se = rainbow_std / np.sqrt(n_simulations)
            
            # Results
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Option Price", f"${rainbow_price:.4f}")
            result_col2.metric("Std Error", f"${rainbow_se:.4f}")
            result_col3.metric("Avg Asset 1", f"${np.mean(final_prices1):.2f}")
            result_col4.metric("Avg Asset 2", f"${np.mean(final_prices2):.2f}")
            
            # Visualization
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Price Paths", "Asset Correlation", "Payoff Analysis"])
            
            with viz_tab1:
                fig_rainbow = go.Figure()
                
                num_display = min(30, n_simulations)
                time_steps = np.linspace(0, T, n_steps + 1)
                
                for i in range(num_display):
                    fig_rainbow.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths1[i],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        opacity=0.4,
                        showlegend=False
                    ))
                    
                    fig_rainbow.add_trace(go.Scatter(
                        x=time_steps,
                        y=paths2[i],
                        mode='lines',
                        line=dict(color='red', width=1),
                        opacity=0.4,
                        showlegend=False
                    ))
                
                fig_rainbow.add_hline(y=K, line_dash="dash", line_color="black",
                                     annotation_text=f"Strike: ${K:.2f}")
                
                fig_rainbow.update_layout(
                    title="Rainbow Option: Asset Price Paths (Blue=Asset1, Red=Asset2)",
                    xaxis_title="Time (Years)",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig_rainbow, use_container_width=True)
            
            with viz_tab2:
                # Scatter plot showing correlation
                fig_corr = go.Figure()
                
                fig_corr.add_trace(go.Scatter(
                    x=final_prices1[:1000],
                    y=final_prices2[:1000],
                    mode='markers',
                    marker=dict(size=5, color=payoffs[:1000], colorscale='Viridis',
                              showscale=True, colorbar=dict(title="Payoff ($)")),
                    showlegend=False
                ))
                
                # Add reference lines
                fig_corr.add_hline(y=np.mean(final_prices2), line_dash="dot", line_color="red")
                fig_corr.add_vline(x=np.mean(final_prices1), line_dash="dot", line_color="blue")
                
                # Calculate realized correlation
                realized_corr = np.corrcoef(final_prices1, final_prices2)[0, 1]
                
                fig_corr.update_layout(
                    title=f"Asset Correlation at Maturity (Realized: {realized_corr:.3f})",
                    xaxis_title="Asset 1 Final Price ($)",
                    yaxis_title="Asset 2 Final Price ($)",
                    height=500
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                corr_col1, corr_col2 = st.columns(2)
                corr_col1.metric("Target Correlation", f"{correlation:.3f}")
                corr_col2.metric("Realized Correlation", f"{realized_corr:.3f}")
            
            with viz_tab3:
                fig_payoff = make_subplots(rows=1, cols=2,
                                          subplot_titles=("Payoff Distribution", "Payoff Heatmap"))
                
                # Histogram of payoffs
                fig_payoff.add_trace(go.Histogram(
                    x=payoffs,
                    nbinsx=50,
                    marker_color='purple',
                    showlegend=False
                ), row=1, col=1)
                
                # 2D histogram/heatmap
                fig_payoff.add_trace(go.Histogram2d(
                    x=final_prices1,
                    y=final_prices2,
                    colorscale='Viridis',
                    showscale=False
                ), row=1, col=2)
                
                fig_payoff.update_xaxes(title_text="Payoff ($)", row=1, col=1)
                fig_payoff.update_yaxes(title_text="Frequency", row=1, col=1)
                fig_payoff.update_xaxes(title_text="Asset 1 Price ($)", row=1, col=2)
                fig_payoff.update_yaxes(title_text="Asset 2 Price ($)", row=1, col=2)
                
                fig_payoff.update_layout(height=400)
                st.plotly_chart(fig_payoff, use_container_width=True)
                
                # Statistics
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Prob ITM", f"{np.mean(payoffs > 0):.2%}")
                stat_col2.metric("Avg Payoff (if ITM)", f"${np.mean(payoffs[payoffs > 0]):.2f}" if np.any(payoffs > 0) else "N/A")
                stat_col3.metric("Max Payoff", f"${np.max(payoffs):.2f}")
    
    # === COMPARISON WITH VANILLA OPTION ===
    st.markdown("---")
    st.subheader("📊 Comparison with Vanilla Option")
    
    # Calculate vanilla option price for comparison
    from scipy.stats import norm
    
    d1_vanilla = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2_vanilla = d1_vanilla - sigma*np.sqrt(T)
    
    if option_type == "call":
        vanilla_price = S*np.exp(-q*T)*norm.cdf(d1_vanilla) - K*np.exp(-r*T)*norm.cdf(d2_vanilla)
    else:
        vanilla_price = K*np.exp(-r*T)*norm.cdf(-d2_vanilla) - S*np.exp(-q*T)*norm.cdf(-d1_vanilla)
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    # Get exotic price based on type
    if exotic_type == "Asian Options":
        exotic_price_display = asian_price
    elif exotic_type == "Barrier Options":
        exotic_price_display = barrier_price
    elif exotic_type == "Lookback Options":
        exotic_price_display = lookback_price
    elif exotic_type == "Binary/Digital Options":
        exotic_price_display = binary_price
    elif exotic_type == "Chooser Options":
        exotic_price_display = chooser_price
    else:  # Rainbow
        exotic_price_display = rainbow_price
    
    comp_col1.metric("Vanilla Option Price", f"${vanilla_price:.4f}")
    comp_col2.metric(f"{exotic_type} Price", f"${exotic_price_display:.4f}")
    
    price_diff = exotic_price_display - vanilla_price
    price_diff_pct = (price_diff / vanilla_price * 100) if vanilla_price > 0 else 0
    
    comp_col3.metric("Difference", f"${price_diff:+.4f}", f"{price_diff_pct:+.1f}%")
    
    # Explanation
    with st.expander("💡 Why the Price Difference?"):
        st.markdown(f"""
        **{exotic_type}** vs **Vanilla {option_type.title()}**:
        
        The price difference reflects the unique features of exotic options:
        
        - **Asian Options**: Typically cheaper due to averaging reducing volatility
        - **Barrier Options**: Usually cheaper (knock-out) or more expensive (knock-in) depending on barrier proximity
        - **Lookback Options**: More expensive due to optimal exercise feature
        - **Binary Options**: Different payoff structure makes direct comparison complex
        - **Chooser Options**: More expensive due to flexibility to choose call or put
        - **Rainbow Options**: Price depends on correlation and which asset performs better/worse
        
        **Current Difference:** ${price_diff:+.4f} ({price_diff_pct:+.1f}%)
        """)
    
    # === EDUCATIONAL SECTION ===
    st.markdown("---")
    with st.expander("📚 Learn More About Exotic Options"):
        st.markdown("""
        ### What Are Exotic Options?
        
        Exotic options are derivatives with more complex features than standard vanilla options. They offer:
        - **Customization** for specific hedging needs
        - **Cost efficiency** for targeted exposures
        - **Enhanced returns** through structured payoffs
        
        ### Common Applications:
        
        **Asian Options:**
        - Commodity hedging (oil, metals)
        - FX forwards with average rates
        - Reducing manipulation risk in thinly traded assets
        
        **Barrier Options:**
        - Reducing premium costs with knock-outs
        - Structured products
        - Currency hedging with risk limits
        
        **Lookback Options:**
        - Eliminating timing risk
        - Performance-based compensation
        - Maximizing/minimizing prices over period
        
        **Binary Options:**
        - Event-driven strategies
        - All-or-nothing bets
        - Short-term speculation
        
        **Chooser Options:**
        - Uncertain market direction
        - Pre-earnings flexibility
        - Volatility trading
        
        **Rainbow Options:**
        - Multi-asset portfolios
        - Worst-of/best-of structures
        - Correlation trading
        
        ### ⚠️ Risks:
        - More complex to understand and value
        - Less liquid than vanilla options
        - Model risk in pricing
        - Path dependency can lead to unexpected outcomes
        """)

if __name__ == "__main__":
    exotic_options_page()
