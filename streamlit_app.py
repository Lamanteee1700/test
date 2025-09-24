import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils import bs_price, greeks, d1, d2

def enhanced_options_page():
    st.title("🎯 Black-Scholes Option Pricer with Advanced Analytics")
    
    # Add market context banner
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px; border-left: 5px solid #1f77b4;">
    <strong>Options Pricing Dashboard:</strong> Analyze European options using the Black-Scholes model with real-time data and comprehensive risk metrics.
    </div>
    """, unsafe_allow_html=True)
    
    # === ENHANCED INPUT SECTION ===
    with st.container():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📈 Market Data")
            
            # Ticker input with validation
            ticker_col1, ticker_col2 = st.columns([3, 1])
            with ticker_col1:
                ticker = st.text_input("Stock Ticker", "AAPL", help="Enter any Yahoo Finance ticker symbol")
            with ticker_col2:
                refresh_data = st.button("🔄", help="Refresh data")
            
            # Fetch and display stock data with error handling
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="5d")
                info = stock.info
                
                if data.empty:
                    st.error("Invalid ticker or no data available")
                    return
                    
                S = data["Close"].iloc[-1]
                prev_close = data["Close"].iloc[-2] if len(data) > 1 else S
                change_pct = (S - prev_close) / prev_close * 100
                
                # Enhanced price display
                price_col1, price_col2, price_col3 = st.columns(3)
                price_col1.metric("Current Price", f"${S:.2f}", f"{change_pct:+.2f}%")
                
                # Add 52-week range if available
                if 'fiftyTwoWeekHigh' in info and 'fiftyTwoWeekLow' in info:
                    high_52w = info['fiftyTwoWeekHigh']
                    low_52w = info['fiftyTwoWeekLow']
                    price_col2.metric("52W High", f"${high_52w:.2f}")
                    price_col3.metric("52W Low", f"${low_52w:.2f}")
                
                # Historical volatility calculation
                if len(data) >= 20:
                    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                    hist_vol = returns.std() * np.sqrt(252)
                    st.info(f"📊 Historical Volatility (20-day): {hist_vol:.1%}")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                S = 100.0  # Default fallback
                st.warning("Using default price of $100 for demonstration")
        
        with col2:
            st.subheader("⚙️ Option Parameters")
            
            # Enhanced parameter inputs
            option_type = st.selectbox("Option Type", ["call", "put"], 
                                     help="Call options profit when price rises, puts when price falls")
            
            # Time to expiration with calendar
            expiry_method = st.radio("Expiration Method", ["Days", "Calendar Date"], horizontal=True)
            
            if expiry_method == "Days":
                T_days = st.number_input("Time to Expiry (days)", value=30, min_value=1, max_value=1000,
                                       help="Trading days until option expires")
                T = T_days / 365
            else:
                expiry_date = st.date_input("Expiration Date", 
                                          value=pd.Timestamp.now() + pd.Timedelta(days=30))
                T_days = (expiry_date - pd.Timestamp.now().date()).days
                T = T_days / 365
                st.info(f"Days to expiry: {T_days}")
            
            # Market environment inputs
            r = st.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.25,
                         help="Treasury rate for similar maturity") / 100
            
            sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0, 2.5,
                            help="Annual volatility (standard deviation of returns)") / 100
    
    # === ENHANCED STRIKE SELECTION ===
    st.subheader("🎯 Strike Price Selection")
    
    strike_col1, strike_col2 = st.columns([2, 1])
    
    with strike_col1:
        strike_method = st.radio(
            "Strike Selection Method",
            ["Preset Levels", "Custom Strike", "Moneyness %"],
            horizontal=True,
            help="Choose how to set the option strike price"
        )
        
        if strike_method == "Preset Levels":
            preset_options = {
                "Deep ITM": S * (0.85 if option_type == "call" else 1.15),
                "ITM": S * (0.95 if option_type == "call" else 1.05),
                "ATM": S,
                "OTM": S * (1.05 if option_type == "call" else 0.95),
                "Deep OTM": S * (1.15 if option_type == "call" else 0.85)
            }
            
            preset_choice = st.selectbox("Select Strike Level", list(preset_options.keys()), index=2)
            K = preset_options[preset_choice]
            
        elif strike_method == "Custom Strike":
            K = st.number_input("Strike Price ($)", value=float(S), min_value=0.01, step=0.01)
            
        else:  # Moneyness %
            moneyness_pct = st.slider("Moneyness (%)", 70, 130, 100, 1,
                                    help="Strike as % of current price")
            K = S * (moneyness_pct / 100)
    
    with strike_col2:
        # Moneyness indicator
        moneyness = S / K
        
        if option_type == "call":
            if moneyness > 1.05:
                status = "🟢 In-The-Money"
                color = "green"
            elif moneyness > 0.95:
                status = "🔵 At-The-Money"
                color = "blue"
            else:
                status = "🔴 Out-Of-The-Money"
                color = "red"
        else:
            if moneyness < 0.95:
                status = "🟢 In-The-Money"
                color = "green"
            elif moneyness < 1.05:
                status = "🔵 At-The-Money"
                color = "blue"
            else:
                status = "🔴 Out-Of-The-Money"
                color = "red"
        
        st.markdown(f"**Selected Strike:** ${K:.2f}")
        st.markdown(f"**Status:** {status}")
        st.markdown(f"**Moneyness:** {moneyness:.3f}")
    
    # === OPTION VALUATION & GREEKS ===
    try:
        price = bs_price(S, K, T, r, sigma, option=option_type)
        delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option=option_type)
        
        intrinsic = max(0, (S-K) if option_type=='call' else (K-S))
        time_value = max(0, price - intrinsic)
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return
    
    # === ENHANCED RESULTS DISPLAY ===
    st.subheader("💰 Valuation Results")
    
    # Main metrics in enhanced layout
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.metric("Option Price", f"${price:.3f}", 
                 help=f"Theoretical {option_type} option value")
    
    with metric_cols[1]:
        st.metric("Intrinsic Value", f"${intrinsic:.3f}",
                 help="Value if exercised immediately")
    
    with metric_cols[2]:
        st.metric("Time Value", f"${time_value:.3f}",
                 help="Premium above intrinsic value")
    
    with metric_cols[3]:
        leverage = (delta * S) / price if price > 0.001 else 0
        st.metric("Leverage", f"{leverage:.1f}x",
                 help="Effective leverage vs. owning stock")
    
    with metric_cols[4]:
        breakeven = K + price if option_type == 'call' else K - price
        st.metric("Breakeven", f"${breakeven:.2f}",
                 help="Stock price needed to break even")
    
    # === ENHANCED GREEKS DASHBOARD ===
    st.subheader("📊 Greeks Risk Analytics")
    
    # Create enhanced Greeks display with interpretations
    greeks_data = {
        "Greek": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
        "Value": [f"{delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{theta:.4f}", f"{rho:.4f}"],
        "Interpretation": [
            f"${delta*100:.0f} profit per $100 stock move",
            f"Delta changes by {gamma:.4f} per $1 stock move", 
            f"${vega*100:.0f} profit per 1% volatility increase",
            f"${theta:.2f} daily time decay",
            f"${rho*100:.0f} profit per 1% rate increase"
        ],
        "Risk Level": [
            "High" if abs(delta) > 0.7 else "Medium" if abs(delta) > 0.3 else "Low",
            "High" if gamma > 0.05 else "Medium" if gamma > 0.02 else "Low",
            "High" if abs(vega) > 15 else "Medium" if abs(vega) > 5 else "Low",
            "High" if abs(theta) > 5 else "Medium" if abs(theta) > 1 else "Low",
            "High" if abs(rho) > 10 else "Medium" if abs(rho) > 3 else "Low"
        ]
    }
    
    greeks_df = pd.DataFrame(greeks_data)
    
    # Display Greeks table with color coding
    st.dataframe(
        greeks_df,
        use_container_width=True,
        column_config={
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Risk Level": st.column_config.TextColumn(
                "Risk Level",
                width="small"
            )
        }
    )
    
    # === ADVANCED VISUALIZATIONS ===
    st.subheader("📈 Advanced Analytics & Visualizations")
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Greeks Surface", "💹 P&L Analysis", "⏰ Time Decay", "🌊 Volatility Impact"
    ])
    
    with viz_tab1:
        st.write("**Greeks Sensitivity Across Price Range**")
        
        # Price range for analysis
        price_range_pct = st.slider("Price Range (±%)", 10, 50, 30, 5)
        price_min = S * (1 - price_range_pct/100)
        price_max = S * (1 + price_range_pct/100)
        
        S_range = np.linspace(price_min, price_max, 100)
        
        # Calculate Greeks across price range
        greeks_arrays = {
            'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Price': []
        }
        
        for S_val in S_range:
            try:
                opt_price = bs_price(S_val, K, T, r, sigma, option_type)
                d, g, v, t, rho_val = greeks(S_val, K, T, r, sigma, option_type)
                
                greeks_arrays['Delta'].append(d)
                greeks_arrays['Gamma'].append(g)
                greeks_arrays['Vega'].append(v)
                greeks_arrays['Theta'].append(t)
                greeks_arrays['Price'].append(opt_price)
                
            except:
                greeks_arrays['Delta'].append(np.nan)
                greeks_arrays['Gamma'].append(np.nan)
                greeks_arrays['Vega'].append(np.nan)
                greeks_arrays['Theta'].append(np.nan)
                greeks_arrays['Price'].append(np.nan)
        
        # Create subplots for Greeks
        fig_greeks = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta'),
            vertical_spacing=0.1
        )
        
        # Add traces
        fig_greeks.add_trace(go.Scatter(x=S_range, y=greeks_arrays['Delta'], 
                                       name='Delta', line=dict(color='blue', width=2)),
                            row=1, col=1)
        fig_greeks.add_trace(go.Scatter(x=S_range, y=greeks_arrays['Gamma'],
                                       name='Gamma', line=dict(color='red', width=2)),
                            row=1, col=2)
        fig_greeks.add_trace(go.Scatter(x=S_range, y=greeks_arrays['Vega'],
                                       name='Vega', line=dict(color='green', width=2)),
                            row=2, col=1)
        fig_greeks.add_trace(go.Scatter(x=S_range, y=greeks_arrays['Theta'],
                                       name='Theta', line=dict(color='purple', width=2)),
                            row=2, col=2)
        
        # Add current price line to all subplots
        for row in [1, 2]:
            for col in [1, 2]:
                fig_greeks.add_vline(x=S, line_dash="dash", line_color="black", 
                                   opacity=0.5, row=row, col=col)
                fig_greeks.add_vline(x=K, line_dash="dot", line_color="red",
                                   opacity=0.5, row=row, col=col)
        
        fig_greeks.update_layout(height=600, showlegend=False,
                               title_text="Greeks Sensitivity Analysis")
        fig_greeks.update_xaxes(title_text="Stock Price ($)")
        
        st.plotly_chart(fig_greeks, use_container_width=True)
    
    with viz_tab2:
        st.write("**Profit & Loss Analysis at Expiration**")
        
        # P&L calculation
        S_expiry = np.linspace(S*0.7, S*1.3, 100)
        option_pnl = []
        stock_pnl = []
        
        for S_exp in S_expiry:
            # Option P&L (assuming we bought the option)
            if option_type == 'call':
                option_value_exp = max(0, S_exp - K)
            else:
                option_value_exp = max(0, K - S_exp)
            
            opt_pnl = option_value_exp - price
            stock_pnl_val = S_exp - S
            
            option_pnl.append(opt_pnl)
            stock_pnl.append(stock_pnl_val)
        
        # Create P&L chart
        fig_pnl = go.Figure()
        
        fig_pnl.add_trace(go.Scatter(
            x=S_expiry, y=option_pnl,
            name=f'{option_type.title()} Option P&L',
            line=dict(color='blue', width=3)
        ))
        
        fig_pnl.add_trace(go.Scatter(
            x=S_expiry, y=stock_pnl,
            name='Stock P&L',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Add reference lines
        fig_pnl.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
        fig_pnl.add_vline(x=S, line_dash="dash", line_color="black", opacity=0.5,
                         annotation_text="Current Price")
        fig_pnl.add_vline(x=breakeven, line_dash="dot", line_color="red",
                         annotation_text=f"Breakeven: ${breakeven:.0f}")
        
        fig_pnl.update_layout(
            title="Profit/Loss at Expiration",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # P&L statistics
        max_profit_idx = np.argmax(option_pnl)
        max_loss_idx = np.argmin(option_pnl)
        
        pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
        pnl_col1.metric("Max Profit", f"${max(option_pnl):.2f}" if max(option_pnl) < 1000 else "Unlimited")
        pnl_col2.metric("Max Loss", f"${min(option_pnl):.2f}")
        pnl_col3.metric("Prob. of Profit", f"{np.mean(np.array(option_pnl) > 0)*100:.1f}%")
    
    with viz_tab3:
        st.write("**Time Decay Analysis**")
        
        # Time decay over remaining life
        days_range = np.linspace(T_days, 1, 50)
        time_decay_values = []
        
        for days in days_range:
            T_temp = days / 365
            price_temp = bs_price(S, K, T_temp, r, sigma, option_type)
            time_decay_values.append(price_temp)
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=days_range, y=time_decay_values,
            mode='lines',
            name='Option Value',
            line=dict(color='red', width=3)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=[T_days], y=[price],
            mode='markers',
            name='Current Value',
            marker=dict(size=10, color='blue')
        ))
        
        fig_time.update_layout(
            title="Time Decay Profile",
            xaxis_title="Days to Expiration",
            yaxis_title="Option Value ($)",
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Time decay metrics
        price_1day = bs_price(S, K, (T_days-1)/365, r, sigma, option_type) if T_days > 1 else 0
        price_7day = bs_price(S, K, (T_days-7)/365, r, sigma, option_type) if T_days > 7 else 0
        
        decay_col1, decay_col2 = st.columns(2)
        decay_col1.metric("1-Day Theta", f"${price_1day - price:.3f}")
        decay_col2.metric("7-Day Decay", f"${price_7day - price:.3f}")
    
    with viz_tab4:
        st.write("**Volatility Impact Analysis**")
        
        # Volatility sensitivity
        vol_range = np.linspace(0.05, 0.8, 50)
        vol_prices = []
        
        for vol in vol_range:
            vol_price = bs_price(S, K, T, r, vol, option_type)
            vol_prices.append(vol_price)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_range*100, y=vol_prices,
            mode='lines',
            name='Option Value',
            line=dict(color='green', width=3)
        ))
        
        fig_vol.add_trace(go.Scatter(
            x=[sigma*100], y=[price],
            mode='markers',
            name='Current Volatility',
            marker=dict(size=10, color='blue')
        ))
        
        fig_vol.update_layout(
            title="Volatility Sensitivity (Vega Profile)",
            xaxis_title="Volatility (%)",
            yaxis_title="Option Value ($)",
            height=400
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Volatility scenarios
        vol_scenarios = {
            "Low Vol (-5%)": bs_price(S, K, T, r, max(0.01, sigma-0.05), option_type),
            "Current Vol": price,
            "High Vol (+5%)": bs_price(S, K, T, r, sigma+0.05, option_type)
        }
        
        scenario_df = pd.DataFrame([
            {"Scenario": k, "Option Value": f"${v:.3f}", "Change": f"${v-price:+.3f}"}
            for k, v in vol_scenarios.items()
        ])
        
        st.dataframe(scenario_df, use_container_width=True)
    
    # === IMPLIED VOLATILITY SECTION ===
    st.subheader("🔍 Implied Volatility Analysis")
    
    iv_col1, iv_col2 = st.columns(2)
    
    with iv_col1:
        market_price = st.number_input(
            "Market Option Price ($)", 
            min_value=0.0, 
            step=0.01,
            help="Enter observed market price to calculate implied volatility"
        )
        
        if market_price > 0:
            try:
                def option_price_diff(vol):
                    return bs_price(S, K, T, r, vol, option_type) - market_price
                
                implied_vol = brentq(option_price_diff, 0.001, 5.0)
                
                st.success(f"**Implied Volatility: {implied_vol:.2%}**")
                
                # Compare with input volatility
                vol_diff = implied_vol - sigma
                if abs(vol_diff) > 0.02:  # 2% difference threshold
                    if vol_diff > 0:
                        st.warning(f"⚠️ IV is {vol_diff:.2%} higher than assumed volatility")
                        st.write("**Interpretation:** Option may be expensive relative to expected volatility")
                    else:
                        st.info(f"ℹ️ IV is {abs(vol_diff):.2%} lower than assumed volatility")
                        st.write("**Interpretation:** Option may be cheap relative to expected volatility")
                else:
                    st.success("✅ IV is close to assumed volatility - fair pricing indicated")
                        
            except (ValueError, RuntimeError):
                st.error("❌ Could not calculate implied volatility. Check if market price is reasonable.")
    
    with iv_col2:
        if market_price > 0 and 'implied_vol' in locals():
            # IV vs Historical Vol comparison
            try:
                if 'hist_vol' in locals():
                    iv_hist_ratio = implied_vol / hist_vol
                    
                    st.metric("IV vs Historical Vol", f"{iv_hist_ratio:.2f}x")
                    
                    if iv_hist_ratio > 1.2:
                        st.warning("🔺 IV significantly above historical - high premium")
                    elif iv_hist_ratio < 0.8:
                        st.info("🔻 IV below historical - potentially undervalued")
                    else:
                        st.success("⚖️ IV in line with historical volatility")
            except:
                pass
    
    # === EDUCATIONAL SUMMARY ===
    with st.expander("📚 Options Trading Insights & Key Concepts"):
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            **🎯 Key Takeaways for This Option:**
            """)
            
            # Dynamic insights based on current option characteristics
            insights = []
            
            if abs(delta) > 0.7:
                insights.append(f"• **High Delta ({delta:.2f}):** Acts like {abs(delta):.0%} of stock position")
            
            if gamma > 0.05:
                insights.append(f"• **High Gamma:** Delta changes rapidly - position risk accelerates near strike")
            
            if abs(theta) > 3:
                insights.append(f"• **Significant Time Decay:** Losing ${abs(theta):.2f}/day - time is critical")
            
            if time_value / price > 0.7 if price > 0.01 else False:
                insights.append(f"• **High Time Value:** {time_value/price:.0%} of price is time premium")
            
            if abs(vega) > 10:
                insights.append(f"• **Volatility Sensitive:** ±$1 change per 1% volatility move")
            
            for insight in insights:
                st.markdown(insight)
        
        with insights_col2:
            st.markdown("""
            **📖 Black-Scholes Model Assumptions:**
            - European exercise (only at expiration)
            - Constant volatility and interest rates  
            - No dividends during option life
            - Continuous trading with no gaps
            - Log-normal stock price distribution
            
            **⚠️ Real-World Considerations:**
            - Bid-ask spreads and transaction costs
            - Early exercise features (American options)
            - Dividend adjustments
            - Volatility smile/skew effects
            - Interest rate and yield curve risks
            """)

if __name__ == "__main__":
    enhanced_options_page()
