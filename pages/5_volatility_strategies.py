# pages/5_volatility_strategies.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils import bs_price, greeks
from datetime import datetime, timedelta

def calculate_historical_volatility(prices, window_days=20):
    """Calculate historical volatility from price series"""
    returns = np.log(prices / prices.shift(1)).dropna()
    rolling_std = returns.rolling(window=window_days).std()
    annualized_vol = rolling_std * np.sqrt(252)  # Annualize assuming 252 trading days
    return returns, annualized_vol

def calculate_garch_volatility(returns, window=252):
    """Simplified GARCH(1,1) volatility estimation"""
    # Simple GARCH approximation - in practice would use more sophisticated methods
    squared_returns = returns**2
    long_run_var = squared_returns.mean()
    
    # GARCH parameters (simplified)
    alpha = 0.1  # Reaction to recent shocks
    beta = 0.85   # Persistence of volatility
    omega = long_run_var * (1 - alpha - beta)
    
    garch_var = [long_run_var]
    for i in range(1, len(squared_returns)):
        var_forecast = omega + alpha * squared_returns.iloc[i-1] + beta * garch_var[-1]
        garch_var.append(var_forecast)
    
    garch_vol = np.sqrt(np.array(garch_var) * 252)  # Annualize
    return garch_vol

def create_volatility_smile(S, T, r, strikes_pct=None, vol_base=0.2):
    """Create a realistic volatility smile/skew"""
    if strikes_pct is None:
        strikes_pct = np.linspace(0.8, 1.2, 21)  # 80% to 120% of spot
    
    strikes = S * strikes_pct
    
    # Create volatility smile with realistic skew
    moneyness = np.log(strikes / S)
    
    # Volatility smile parameters
    atm_vol = vol_base
    skew = -0.1  # Negative skew (puts more expensive)
    smile = 0.02  # Smile curvature
    
    implied_vols = atm_vol + skew * moneyness + smile * moneyness**2
    
    return strikes, implied_vols

def volatility_cone_analysis(ticker, periods=[10, 20, 30, 60, 90, 120]):
    """Calculate volatility cone for different time periods"""
    try:
        # Fetch longer history for volatility cone
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="2y")
        
        if hist_data.empty:
            return None
            
        prices = hist_data['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        cone_data = []
        
        for period in periods:
            if len(returns) > period:
                rolling_vol = returns.rolling(window=period).std() * np.sqrt(252)
                rolling_vol = rolling_vol.dropna()
                
                if len(rolling_vol) > 0:
                    cone_data.append({
                        'Period': period,
                        'Current': rolling_vol.iloc[-1],
                        'Min': rolling_vol.min(),
                        'Max': rolling_vol.max(),
                        'Mean': rolling_vol.mean(),
                        'Percentile_25': rolling_vol.quantile(0.25),
                        'Percentile_75': rolling_vol.quantile(0.75)
                    })
        
        return pd.DataFrame(cone_data)
        
    except Exception as e:
        st.error(f"Error calculating volatility cone: {str(e)}")
        return None

def long_straddle_analysis(S, K, T, r, sigma, premium_paid):
    """Analyze long straddle strategy"""
    
    # Calculate individual option prices
    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    theoretical_premium = call_price + put_price
    
    # Price range for analysis
    price_range = np.linspace(S * 0.6, S * 1.4, 100)
    
    # Calculate P&L at expiration
    straddle_payoff = []
    for S_exp in price_range:
        call_payoff = max(0, S_exp - K)
        put_payoff = max(0, K - S_exp)
        total_payoff = call_payoff + put_payoff - premium_paid
        straddle_payoff.append(total_payoff)
    
    # Calculate breakeven points
    upper_breakeven = K + premium_paid
    lower_breakeven = K - premium_paid
    
    # Calculate Greeks for the straddle
    call_greeks = greeks(S, K, T, r, sigma, "call")
    put_greeks = greeks(S, K, T, r, sigma, "put")
    
    straddle_greeks = {
        'delta': call_greeks[0] + put_greeks[0],  # Should be close to 0 for ATM
        'gamma': call_greeks[1] + put_greeks[1],  # Positive
        'vega': call_greeks[2] + put_greeks[2],   # Positive
        'theta': call_greeks[3] + put_greeks[3],  # Negative
        'rho': call_greeks[4] + put_greeks[4]
    }
    
    return {
        'price_range': price_range,
        'payoff': straddle_payoff,
        'upper_breakeven': upper_breakeven,
        'lower_breakeven': lower_breakeven,
        'theoretical_premium': theoretical_premium,
        'greeks': straddle_greeks
    }

def vol_trading_simulator(initial_vol, target_vol, days_to_expiry, S, K, r):
    """Simulate P&L from volatility trading"""
    
    # Create time series
    time_steps = np.linspace(days_to_expiry, 1, days_to_expiry) / 365
    
    # Simulate volatility path (mean-reverting)
    vol_path = [initial_vol]
    mean_reversion_speed = 2.0  # How fast vol reverts to long-term mean
    vol_of_vol = 0.3  # Volatility of volatility
    long_term_vol = target_vol
    
    dt = 1/365
    
    for i in range(len(time_steps) - 1):
        # Mean-reverting volatility model
        dvol = mean_reversion_speed * (long_term_vol - vol_path[-1]) * dt + \
               vol_of_vol * vol_path[-1] * np.random.normal(0, np.sqrt(dt))
        new_vol = max(0.05, vol_path[-1] + dvol)  # Floor volatility at 5%
        vol_path.append(new_vol)
    
    # Calculate option values along the path
    option_values = []
    for i, (t, vol) in enumerate(zip(time_steps, vol_path)):
        if t > 0:
            call_val = bs_price(S, K, t, r, vol, "call")
            put_val = bs_price(S, K, t, r, vol, "put")
            straddle_val = call_val + put_val
            option_values.append(straddle_val)
        else:
            # At expiration
            call_payoff = max(0, S - K)
            put_payoff = max(0, K - S)
            option_values.append(call_payoff + put_payoff)
    
    return time_steps[:-1], vol_path[:-1], option_values

def show_volatility_strategies_page():
    st.set_page_config(page_title="Volatility Strategies", layout="wide")
    st.title("🌊 Advanced Volatility Strategies & Analysis")

    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Comprehensive Volatility Trading Framework</h3>
        <p style="color: #e6f3ff; margin: 0.5rem 0 0 0;">
            Master volatility trading through historical analysis, Greeks understanding, and strategy simulation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === SECTION 1: REAL-TIME VOLATILITY ANALYSIS ===
    st.subheader("📊 Real-Time Volatility Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Stock selection
        ticker = st.text_input("Stock Ticker", "AAPL", help="Enter any ticker (e.g., AAPL, MSFT, GOOGL)")
        vol_window = st.slider("Volatility Window (days)", 10, 60, 20, help="Rolling window for volatility calculation")
        
    with col2:
        # VIX Analysis
        st.markdown("**📈 VIX Fear & Greed Index**")
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                prev_vix = vix_data['Close'].iloc[-2] if len(vix_data) > 1 else current_vix
                vix_change = current_vix - prev_vix
                
                st.metric("Current VIX", f"{current_vix:.1f}", f"{vix_change:+.1f}")
                
                # VIX interpretation
                if current_vix < 15:
                    vix_regime = "🟢 Low Volatility (Complacency)"
                elif current_vix < 25:
                    vix_regime = "🟡 Normal Volatility"
                elif current_vix < 35:
                    vix_regime = "🟠 Elevated Volatility (Caution)"
                else:
                    vix_regime = "🔴 High Volatility (Fear)"
                    
                st.info(vix_regime)
            else:
                st.warning("VIX data unavailable")
        except:
            st.warning("Could not fetch VIX data")

    # Fetch and analyze stock data
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="6mo")  # 6 months for better volatility analysis
        
        if hist_data.empty:
            st.error("No data available for this ticker")
            return
            
        current_price = hist_data['Close'].iloc[-1]
        st.success(f"**{ticker}** Current Price: ${current_price:.2f}")
        
        # Calculate volatilities
        returns, hist_vol = calculate_historical_volatility(hist_data['Close'], vol_window)
        garch_vol = calculate_garch_volatility(returns.dropna())
        
        # Current volatility metrics
        current_hist_vol = hist_vol.iloc[-1] if not hist_vol.empty else 0.2
        current_garch_vol = garch_vol[-1] if len(garch_vol) > 0 else 0.2
        
        # Volatility metrics display
        vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
        vol_col1.metric("Historical Vol", f"{current_hist_vol:.1%}")
        vol_col2.metric("GARCH Vol", f"{current_garch_vol:.1%}")
        vol_col3.metric("Realized Vol (Ann.)", f"{returns.std() * np.sqrt(252):.1%}")
        vol_col4.metric("Vol Regime", 
                       "High" if current_hist_vol > 0.3 else "Normal" if current_hist_vol > 0.15 else "Low")
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return

    # === SECTION 2: VOLATILITY VISUALIZATIONS ===
    st.subheader("📈 Volatility Analysis & Visualizations")
    
    viz_tabs = st.tabs(["📊 Vol History", "🎯 Vol Cone", "😊 Vol Smile", "🌊 Vol Surface"])
    
    with viz_tabs[0]:
        st.markdown("**Historical Volatility Evolution**")
        
        # Create volatility chart
        fig_vol = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price History', 'Volatility History'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Price chart
        fig_vol.add_trace(
            go.Scatter(x=hist_data.index, y=hist_data['Close'], 
                      name=f'{ticker} Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volatility chart
        if not hist_vol.empty:
            fig_vol.add_trace(
                go.Scatter(x=hist_vol.index, y=hist_vol*100, 
                          name=f'{vol_window}D Historical Vol', line=dict(color='red')),
                row=2, col=1
            )
        
        # Add GARCH volatility if available
        if len(garch_vol) > 0:
            garch_dates = hist_data.index[-len(garch_vol):]
            fig_vol.add_trace(
                go.Scatter(x=garch_dates, y=garch_vol*100, 
                          name='GARCH Vol', line=dict(color='green', dash='dash')),
                row=2, col=1
            )
        
        fig_vol.update_layout(height=600, showlegend=True)
        fig_vol.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_vol.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("**Volatility Cone Analysis**")
        
        cone_data = volatility_cone_analysis(ticker)
        
        if cone_data is not None and not cone_data.empty:
            # Create volatility cone chart
            fig_cone = go.Figure()
            
            # Add cone bands
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Max']*100,
                mode='lines', name='Max', line=dict(color='red', width=1)
            ))
            
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Percentile_75']*100,
                mode='lines', name='75th Percentile', line=dict(color='orange', width=2)
            ))
            
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Mean']*100,
                mode='lines+markers', name='Mean', line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Percentile_25']*100,
                mode='lines', name='25th Percentile', line=dict(color='orange', width=2)
            ))
            
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Min']*100,
                mode='lines', name='Min', line=dict(color='red', width=1)
            ))
            
            fig_cone.add_trace(go.Scatter(
                x=cone_data['Period'], y=cone_data['Current']*100,
                mode='markers', name='Current', marker=dict(size=12, color='green', symbol='diamond')
            ))
            
            fig_cone.update_layout(
                title="Volatility Cone - Historical Context",
                xaxis_title="Time Period (Days)",
                yaxis_title="Annualized Volatility (%)",
                height=500
            )
            
            st.plotly_chart(fig_cone, use_container_width=True)
            
            # Display cone table
            cone_display = cone_data.copy()
            for col in ['Current', 'Min', 'Max', 'Mean', 'Percentile_25', 'Percentile_75']:
                cone_display[col] = cone_display[col].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(cone_display, use_container_width=True)
            
        else:
            st.warning("Could not generate volatility cone data")
    
    with viz_tabs[2]:
        st.markdown("**Implied Volatility Smile/Skew**")
        
        # Create volatility smile
        strikes, impl_vols = create_volatility_smile(current_price, 30/365, 0.03, vol_base=current_hist_vol)
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=strikes/current_price*100, y=impl_vols*100,
            mode='lines+markers', name='Implied Volatility',
            line=dict(color='purple', width=3), marker=dict(size=8)
        ))
        
        fig_smile.add_vline(x=100, line_dash="dash", line_color="red", 
                           annotation_text="ATM")
        
        fig_smile.update_layout(
            title="Volatility Smile/Skew Pattern",
            xaxis_title="Moneyness (Strike/Spot %)",
            yaxis_title="Implied Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # Explain the smile
        st.info("""
        **Volatility Smile Interpretation:**
        - **Flat curve**: Market expects normal distribution of returns
        - **Smile shape**: Higher demand for both deep ITM and OTM options
        - **Skew (puts > calls)**: Market fears downside more than upside
        - **Term structure**: How smile changes across expiration dates
        """)
    
    with viz_tabs[3]:
        st.markdown("**3D Volatility Surface**")
        
        # Create 3D volatility surface
        strike_range = np.linspace(0.8, 1.2, 15)
        time_range = np.linspace(7, 365, 12)  # 1 week to 1 year
        
        Strike_mesh, Time_mesh = np.meshgrid(strike_range, time_range)
        Vol_surface = np.zeros_like(Strike_mesh)
        
        # Generate realistic volatility surface
        for i, T_days in enumerate(time_range):
            T_years = T_days / 365
            strikes_abs = strike_range * current_price
            _, vol_smile = create_volatility_smile(current_price, T_years, 0.03, 
                                                  strikes_pct=strike_range, 
                                                  vol_base=current_hist_vol)
            Vol_surface[i, :] = vol_smile
        
        fig_surface = go.Figure(data=[go.Surface(
            z=Vol_surface*100,
            x=Strike_mesh*100,
            y=Time_mesh,
            colorscale='Viridis',
            name='Implied Vol Surface'
        )])
        
        fig_surface.update_layout(
            title="3D Implied Volatility Surface",
            scene=dict(
                xaxis_title='Moneyness (%)',
                yaxis_title='Days to Expiration',
                zaxis_title='Implied Volatility (%)'
            ),
            height=600
        )
        
        st.plotly_chart(fig_surface, use_container_width=True)

    # === SECTION 3: VOLATILITY STRATEGIES ===
    st.subheader("🎯 Volatility Trading Strategies")
    
    strategy_tabs = st.tabs(["🔄 Long Straddle", "📐 Strangles", "🦋 Iron Condor", "⚡ Vol Trading Sim"])
    
    with strategy_tabs[0]:
        st.markdown("**Long Straddle Analysis - Pure Volatility Play**")
        
        straddle_col1, straddle_col2 = st.columns(2)
        
        with straddle_col1:
            st.markdown("**Strategy Parameters**")
            K_straddle = st.number_input("Strike Price", value=current_price, step=0.5)
            T_straddle = st.slider("Days to Expiration", 7, 90, 30) / 365
            r_straddle = st.slider("Risk-free Rate (%)", 0.0, 10.0, 3.0) / 100
            vol_assumption = st.slider("Volatility Assumption (%)", 10.0, 80.0, current_hist_vol*100) / 100
        
        with straddle_col2:
            st.markdown("**Market Conditions**")
            premium_paid = bs_price(current_price, K_straddle, T_straddle, r_straddle, vol_assumption, "call") + \
                          bs_price(current_price, K_straddle, T_straddle, r_straddle, vol_assumption, "put")
            
            st.metric("Total Premium", f"${premium_paid:.2f}")
            st.metric("Breakeven Range", f"${K_straddle - premium_paid:.2f} - ${K_straddle + premium_paid:.2f}")
            st.metric("Required Move", f"±{premium_paid/current_price:.1%}")
        
        # Straddle analysis
        straddle_analysis = long_straddle_analysis(current_price, K_straddle, T_straddle, 
                                                  r_straddle, vol_assumption, premium_paid)
        
        # P&L Chart
        fig_straddle = go.Figure()
        fig_straddle.add_trace(go.Scatter(
            x=straddle_analysis['price_range'], 
            y=straddle_analysis['payoff'],
            mode='lines', name='Straddle P&L',
            line=dict(color='blue', width=3)
        ))
        
        fig_straddle.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_straddle.add_vline(x=current_price, line_dash="dash", line_color="gray", 
                              annotation_text="Current Price")
        fig_straddle.add_vline(x=straddle_analysis['lower_breakeven'], line_dash="dot", 
                              line_color="red", annotation_text="Lower BE")
        fig_straddle.add_vline(x=straddle_analysis['upper_breakeven'], line_dash="dot", 
                              line_color="red", annotation_text="Upper BE")
        
        fig_straddle.update_layout(
            title="Long Straddle Profit/Loss at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            height=400
        )
        
        st.plotly_chart(fig_straddle, use_container_width=True)
        
        # Greeks analysis
        greeks_data = straddle_analysis['greeks']
        st.markdown("**Strategy Greeks**")
        greeks_cols = st.columns(5)
        greeks_cols[0].metric("Delta", f"{greeks_data['delta']:.3f}")
        greeks_cols[1].metric("Gamma", f"{greeks_data['gamma']:.3f}")
        greeks_cols[2].metric("Vega", f"{greeks_data['vega']:.3f}")
        greeks_cols[3].metric("Theta", f"{greeks_data['theta']:.3f}")
        greeks_cols[4].metric("Rho", f"{greeks_data['rho']:.3f}")
    
    with strategy_tabs[1]:
        st.markdown("**Strangle Strategy - Cheaper Volatility Bet**")
        
        strangle_col1, strangle_col2 = st.columns(2)
        
        with strangle_col1:
            call_strike = st.number_input("Call Strike", value=current_price * 1.05, step=0.5)
            put_strike = st.number_input("Put Strike", value=current_price * 0.95, step=0.5)
            T_strangle = st.slider("Days to Exp (Strangle)", 7, 90, 30, key="strangle_t") / 365
            vol_strangle = st.slider("Vol Assumption (Strangle)", 10.0, 80.0, current_hist_vol*100, key="strangle_vol") / 100
        
        # Calculate strangle
        call_premium = bs_price(current_price, call_strike, T_strangle, r_straddle, vol_strangle, "call")
        put_premium = bs_price(current_price, put_strike, T_strangle, r_straddle, vol_strangle, "put")
        total_premium = call_premium + put_premium
        
        with strangle_col2:
            st.metric("Call Premium", f"${call_premium:.2f}")
            st.metric("Put Premium", f"${put_premium:.2f}")
            st.metric("Total Premium", f"${total_premium:.2f}")
            st.metric("Max Loss", f"${total_premium:.2f}")
        
        # Strangle P&L
        price_range = np.linspace(current_price * 0.6, current_price * 1.4, 100)
        strangle_payoff = []
        
        for S_exp in price_range:
            call_payoff = max(0, S_exp - call_strike)
            put_payoff = max(0, put_strike - S_exp)
            net_payoff = call_payoff + put_payoff - total_premium
            strangle_payoff.append(net_payoff)
        
        fig_strangle = go.Figure()
        fig_strangle.add_trace(go.Scatter(
            x=price_range, y=strangle_payoff,
            mode='lines', name='Strangle P&L',
            line=dict(color='green', width=3)
        ))
        
        fig_strangle.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_strangle.add_vline(x=current_price, line_dash="dash", line_color="gray")
        fig_strangle.add_vline(x=put_strike - total_premium, line_dash="dot", line_color="red")
        fig_strangle.add_vline(x=call_strike + total_premium, line_dash="dot", line_color="red")
        
        fig_strangle.update_layout(
            title="Long Strangle Profit/Loss",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            height=400
        )
        
        st.plotly_chart(fig_strangle, use_container_width=True)
    
    with strategy_tabs[2]:
        st.markdown("**Iron Condor - Range-bound Strategy**")
        
        # Iron Condor parameters
        ic_col1, ic_col2 = st.columns(2)
        
        with ic_col1:
            put_strike_short = st.number_input("Short Put Strike", value=current_price * 0.95)
            put_strike_long = st.number_input("Long Put Strike", value=current_price * 0.90)
            
        with ic_col2:
            call_strike_short = st.number_input("Short Call Strike", value=current_price * 1.05)
            call_strike_long = st.number_input("Long Call Strike", value=current_price * 1.10)
        
        T_ic = st.slider("Days to Exp (Iron Condor)", 7, 90, 45, key="ic_days") / 365
        vol_ic = st.slider("Vol Assumption (IC)", 10.0, 80.0, current_hist_vol*100, key="ic_vol") / 100
        
        # Calculate Iron Condor premiums
        put_short_premium = bs_price(current_price, put_strike_short, T_ic, r_straddle, vol_ic, "put")
        put_long_premium = bs_price(current_price, put_strike_long, T_ic, r_straddle, vol_ic, "put")
        call_short_premium = bs_price(current_price, call_strike_short, T_ic, r_straddle, vol_ic, "call")
        call_long_premium = bs_price(current_price, call_strike_long, T_ic, r_straddle, vol_ic, "call")
        
        net_credit = put_short_premium - put_long_premium + call_short_premium - call_long_premium
        max_loss = min(put_strike_short - put_strike_long, call_strike_long - call_strike_short) - net_credit
        
        st.metric("Net Credit Received", f"${net_credit:.2f}")
        st.metric("Maximum Loss", f"${max_loss:.2f}")
        st.metric("Maximum Profit", f"${net_credit:.2f}")
        
        # Iron Condor P&L
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        ic_payoff = []
        
        for S_exp in price_range:
            # Put spread (short - long)
            put_spread = max(0, put_strike_short - S_exp) - max(0, put_strike_long - S_exp)
            # Call spread (short - long)
            call_spread = max(0, S_exp - call_strike_short) - max(0, S_exp - call_strike_long)
            # Net P&L = Credit received - losses on spreads
            net_payoff = net_credit - put_spread - call_spread
            ic_payoff.append(net_payoff)
        
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(
            x=price_range, y=ic_payoff,
            mode='lines', name='Iron Condor P&L',
            line=dict(color='purple', width=3)
        ))
        
        fig_ic.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_ic.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text="Current")
        fig_ic.add_vline(x=put_strike_long, line_dash="dot", line_color="red", annotation_text="Long Put")
        fig_ic.add_vline(x=put_strike_short, line_dash="dot", line_color="orange", annotation_text="Short Put")
        fig_ic.add_vline(x=call_strike_short, line_dash="dot", line_color="orange", annotation_text="Short Call")
        fig_ic.add_vline(x=call_strike_long, line_dash="dot", line_color="red", annotation_text="Long Call")
        
        fig_ic.update_layout(
            title="Iron Condor Profit/Loss Profile",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            height=400
        )
        
        st.plotly_chart(fig_ic, use_container_width=True)
        
        st.info("""
        **Iron Condor Strategy:**
        - **Profit Zone**: Between short strikes (range-bound market)
        - **Max Profit**: Net credit received (if price stays between short strikes)
        - **Max Loss**: Width of spread minus net credit
        - **Best For**: Low volatility, sideways markets
        """)
    
    with strategy_tabs[3]:
        st.markdown("**Volatility Trading Simulator**")
        
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            initial_vol_sim = st.slider("Initial Volatility (%)", 10.0, 60.0, current_hist_vol*100) / 100
            target_vol_sim = st.slider("Target Volatility (%)", 10.0, 60.0, 25.0) / 100
            days_sim = st.slider("Days to Simulate", 10, 90, 30)
            K_sim = st.number_input("Strike for Simulation", value=current_price)
            
        with sim_col2:
            st.markdown("**Simulation Setup**")
            st.write(f"• Trading a straddle at ${K_sim:.2f} strike")
            st.write(f"• Initial vol: {initial_vol_sim:.1%}")
            st.write(f"• Target vol: {target_vol_sim:.1%}")
            st.write(f"• Time horizon: {days_sim} days")
            
            if st.button("🎲 Run Volatility Simulation", use_container_width=True):
                # Run simulation
                time_path, vol_path, option_values = vol_trading_simulator(
                    initial_vol_sim, target_vol_sim, days_sim, current_price, K_sim, r_straddle
                )
                
                # Store results in session state
                st.session_state.sim_results = {
                    'time_path': time_path,
                    'vol_path': vol_path,
                    'option_values': option_values
                }
        
        # Display simulation results
        if 'sim_results' in st.session_state:
            results = st.session_state.sim_results
            
            # Create simulation charts
            fig_sim = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Volatility Path', 'Straddle Value Evolution'),
                vertical_spacing=0.1
            )
            
            # Volatility path
            fig_sim.add_trace(
                go.Scatter(x=list(range(len(results['vol_path']))), y=np.array(results['vol_path'])*100,
                          name='Volatility Path', line=dict(color='red')),
                row=1, col=1
            )
            
            fig_sim.add_hline(y=target_vol_sim*100, line_dash="dash", line_color="blue",
                             annotation_text="Target Vol", row=1, col=1)
            
            # Option values
            fig_sim.add_trace(
                go.Scatter(x=list(range(len(results['option_values']))), y=results['option_values'],
                          name='Straddle Value', line=dict(color='green')),
                row=2, col=1
            )
            
            fig_sim.update_layout(height=600, showlegend=True)
            fig_sim.update_xaxes(title_text="Days", row=2, col=1)
            fig_sim.update_yaxes(title_text="Volatility (%)", row=1, col=1)
            fig_sim.update_yaxes(title_text="Option Value ($)", row=2, col=1)
            
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Simulation statistics
            initial_straddle_value = results['option_values'][0]
            final_straddle_value = results['option_values'][-1]
            total_pnl = final_straddle_value - initial_straddle_value
            vol_change = results['vol_path'][-1] - results['vol_path'][0]
            
            sim_stats_cols = st.columns(4)
            sim_stats_cols[0].metric("Initial Value", f"${initial_straddle_value:.2f}")
            sim_stats_cols[1].metric("Final Value", f"${final_straddle_value:.2f}")
            sim_stats_cols[2].metric("Total P&L", f"${total_pnl:.2f}", f"{total_pnl/initial_straddle_value:.1%}")
            sim_stats_cols[3].metric("Vol Change", f"{vol_change:.1%}")

    # === SECTION 4: EDUCATIONAL CONTENT ===
    st.subheader("📚 Volatility Trading Education")
    
    edu_tabs = st.tabs(["🎯 Key Concepts", "⚖️ Risk Management", "📊 Market Regimes"])
    
    with edu_tabs[0]:
        st.markdown("""
        ### 🎯 Essential Volatility Concepts
        
        **Types of Volatility:**
        - **Historical Volatility**: Actual price movements observed in the past
        - **Implied Volatility**: Market's expectation of future volatility (from option prices)
        - **Realized Volatility**: Actual volatility that occurs over the option's life
        - **GARCH Volatility**: Statistical model accounting for volatility clustering
        
        **Volatility Trading Fundamentals:**
        - **Long Volatility**: Profit when actual volatility > implied volatility
        - **Short Volatility**: Profit when actual volatility < implied volatility
        - **Vega Risk**: Sensitivity to changes in implied volatility
        - **Gamma Scalping**: Dynamic hedging to capture realized volatility
        
        **Key Relationships:**
        - **Vol Smile**: IV varies by strike (skew indicates market sentiment)
        - **Term Structure**: IV varies by expiration (short-term often higher)
        - **Mean Reversion**: Volatility tends to revert to long-term average
        - **Volatility Clustering**: High vol periods followed by high vol periods
        """)
        
        # Add volatility comparison table
        vol_comparison = pd.DataFrame({
            'Strategy': ['Long Straddle', 'Long Strangle', 'Short Iron Condor', 'Calendar Spread'],
            'Volatility View': ['High', 'High', 'Low', 'Vol Term Structure'],
            'Max Risk': ['Premium Paid', 'Premium Paid', 'Spread Width', 'Net Debit'],
            'Max Reward': ['Unlimited', 'Unlimited', 'Net Credit', 'Limited'],
            'Best Market': ['Big moves', 'Big moves', 'Range-bound', 'Time decay + vol']
        })
        
        st.dataframe(vol_comparison, use_container_width=True)
    
    with edu_tabs[1]:
        st.markdown("""
        ### ⚖️ Volatility Risk Management
        
        **Position Sizing:**
        - Never risk more than 2-5% of portfolio on single volatility bet
        - Consider correlation between vol positions
        - Account for time decay (theta) in position sizing
        
        **Greeks Management:**
        - **Delta**: Keep close to neutral for pure vol plays
        - **Gamma**: Monitor for acceleration risk near strikes
        - **Vega**: Primary risk factor - track vol changes closely
        - **Theta**: Time decay enemy of long vol, friend of short vol
        
        **Risk Controls:**
        - **Stop Losses**: Set at 50% of premium paid for long vol
        - **Profit Taking**: Take profits at 100-200% for long vol strategies
        - **Time Management**: Close positions with 1-2 weeks to expiry
        - **Vol Regime Awareness**: Adjust strategy based on vol environment
        
        **Common Mistakes:**
        - Trading volatility without understanding the Greeks
        - Ignoring time decay in strategy selection
        - Not adjusting for changing vol regimes
        - Over-leveraging volatility positions
        """)
    
    with edu_tabs[2]:
        st.markdown("""
        ### 📊 Volatility Regimes & Market Context
        
        **Low Volatility Regime (VIX < 15):**
        - Market complacency, steady uptrends
        - Implied vol often cheaper than realized
        - **Strategy**: Long volatility, short premium strategies
        - **Risk**: Sudden vol spikes, complacency reversals
        
        **Normal Volatility Regime (VIX 15-25):**
        - Balanced market conditions
        - IV generally fair relative to historical levels
        - **Strategy**: Direction-neutral strategies, straddles/strangles
        - **Risk**: Choppy, range-bound markets
        
        **High Volatility Regime (VIX > 25):**
        - Market stress, fear-driven selling
        - Implied vol often expensive vs realized
        - **Strategy**: Short volatility, iron condors, covered calls
        - **Risk**: Continued vol expansion, tail events
        
        **Volatility Clustering:**
        - High vol periods tend to persist
        - Low vol periods also cluster together
        - Important for timing entries/exits
        """)
        
        # Current regime assessment
        if 'current_vix' in locals():
            if current_vix < 15:
                regime_color = "🟢"
                regime_desc = "Low Volatility - Consider long vol strategies"
            elif current_vix < 25:
                regime_color = "🟡" 
                regime_desc = "Normal Volatility - Neutral strategies work well"
            else:
                regime_color = "🔴"
                regime_desc = "High Volatility - Consider short vol strategies"
                
            st.info(f"{regime_color} **Current Regime Assessment:** {regime_desc}")

    # === FOOTER ===
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Advanced Volatility Strategies Dashboard</strong></p>
        <p>Educational tool for understanding volatility trading • Not investment advice</p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# Run the page
if __name__ == "__main__":
    show_volatility_strategies_page()
