import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from utils import bs_price, greeks, d1, d2

def options_page():
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
            
            # Price data source selection
            data_source = st.radio("Data Source", ["Company Search / Ticker", "Create Your Own"], horizontal=True)
            
            if data_source == "Company Search / Ticker":
                # Company name to ticker mapping
                popular_companies = {
                    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Amazon": "AMZN",
                    "Tesla": "TSLA", "Meta": "META", "Netflix": "NFLX", "Nvidia": "NVDA",
                    "JPMorgan": "JPM", "Berkshire": "BRK-B", "Johnson & Johnson": "JNJ",
                    "Visa": "V", "Procter & Gamble": "PG", "Mastercard": "MA", "Disney": "DIS",
                    "Coca Cola": "KO", "McDonald's": "MCD", "Nike": "NKE", "Intel": "INTC",
                    "Walmart": "WMT", "Boeing": "BA", "IBM": "IBM", "Salesforce": "CRM"
                }
                
                company_name = st.selectbox("Select Company or enter ticker below", 
                                          [""] + list(popular_companies.keys()),
                                          help="Choose from popular companies or leave blank to enter ticker")
                
                if company_name:
                    ticker = popular_companies[company_name]
                    st.success(f"Selected: {company_name} ({ticker})")
                else:
                    ticker = st.text_input("Enter ticker symbol:", "AAPL", help="Yahoo Finance ticker symbol")
                
                # Fetch and display stock data with error handling
                try:
                    import yfinance as yf_import
                    stock = yf_import.Ticker(ticker)
                    data = stock.history(period="20d")  # Get more data for volatility calculation
                    info = stock.info
                    
                    if data.empty:
                        st.error("Invalid ticker or no data available")
                        return
                        
                    S = data["Close"].iloc[-1]
                    prev_close = data["Close"].iloc[-2] if len(data) > 1 else S
                    change_pct = (S - prev_close) / prev_close * 100
                    
                    # Calculate historical volatility from data
                    if len(data) >= 10:
                        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                        hist_vol = returns.std() * np.sqrt(252)
                        online_vol_available = True
                    else:
                        hist_vol = 0.2  # Default fallback
                        online_vol_available = False
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    S = 100.0  # Default fallback
                    hist_vol = 0.2
                    online_vol_available = False
                    st.warning("Using default values for demonstration")
            else:
                # Custom price input
                S = st.number_input("Custom Stock Price ($)", value=100.0, min_value=0.01, step=0.01)
                change_pct = 0
                hist_vol = 0.2
                online_vol_available = False
                ticker = "CUSTOM"
                info = None
                st.info(f"Using custom price: ${S:.2f}")
            
            if data_source == "Company Search / Ticker" and info:
                
                # Company Information Display
                if info and len(info) > 5:  # Ensure we have substantial company data
                    company_name = info.get('longName', info.get('shortName', ticker))
                    sector = info.get('sector', 'N/A')
                    industry = info.get('industry', 'N/A')
                    market_cap = info.get('marketCap', 0)
                    
                    # Only display company info if we have meaningful data
                    if company_name and company_name != ticker:
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #007bff; margin: 10px 0;">
                        <h4 style="margin: 0; color: #007bff;">{company_name} ({ticker})</h4>
                        {f'<p style="margin: 5px 0;"><strong>Sector:</strong> {sector}' + (f' | <strong>Industry:</strong> {industry}' if industry != 'N/A' else '') + '</p>' if sector != 'N/A' or industry != 'N/A' else ''}
                        {f'<p style="margin: 5px 0;"><strong>Market Cap:</strong> ${market_cap/1e9:.1f}B</p>' if market_cap > 1e9 else f'<p style="margin: 5px 0;"><strong>Market Cap:</strong> ${market_cap/1e6:.0f}M</p>' if market_cap > 1e6 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add business summary if available (truncated)
                        if 'longBusinessSummary' in info and info['longBusinessSummary']:
                            summary = info['longBusinessSummary']
                            if len(summary) > 250:
                                summary = summary[:250] + "..."
                            
                            with st.expander(f"📋 About {company_name}"):
                                st.write(summary)
                                
                                # Additional company metrics in a clean table
                                metrics_data = []
                                
                                # Core financial metrics
                                if info.get('fullTimeEmployees'):
                                    metrics_data.append({"Metric": "Employees", "Value": f"{info['fullTimeEmployees']:,}"})
                                
                                if info.get('totalRevenue'):
                                    revenue = info['totalRevenue']
                                    if revenue > 1e9:
                                        metrics_data.append({"Metric": "Revenue (TTM)", "Value": f"${revenue/1e9:.1f}B"})
                                    else:
                                        metrics_data.append({"Metric": "Revenue (TTM)", "Value": f"${revenue/1e6:.0f}M"})
                                
                                if info.get('trailingPE') and info['trailingPE'] > 0:
                                    metrics_data.append({"Metric": "P/E Ratio", "Value": f"{info['trailingPE']:.1f}"})
                                
                                if info.get('beta'):
                                    metrics_data.append({"Metric": "Beta", "Value": f"{info['beta']:.2f}"})
                                
                                if info.get('dividendYield'):
                                    metrics_data.append({"Metric": "Dividend Yield", "Value": f"{info['dividendYield']:.2f}%"})

                                
                                if metrics_data:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"📊 Analyzing: **{ticker}** - Limited company information available")
                
                # Enhanced price display
                price_col1, price_col2, price_col3 = st.columns(3)
                price_col1.metric("Current Price", f"${S:.2f}", f"{change_pct:+.2f}%")
                
                # Add 52-week range if available
                if info and 'fiftyTwoWeekHigh' in info and 'fiftyTwoWeekLow' in info:
                    high_52w = info['fiftyTwoWeekHigh']
                    low_52w = info['fiftyTwoWeekLow']
                    price_col2.metric("52W High", f"${high_52w:.2f}")
                    price_col3.metric("52W Low", f"${low_52w:.2f}")
                
                # Historical volatility display
                if online_vol_available:
                    st.info(f"📊 Historical Volatility (20-day): {hist_vol:.1%}")
                elif data_source == "Company Search / Ticker":
                    st.warning("Limited data for volatility calculation")
        
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
            r = st.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.01,
                         help="Treasury rate for similar maturity") / 100
            
            # Volatility input with online data integration
            if data_source == "Company Search / Ticker" and online_vol_available:
                use_hist_vol = st.checkbox(
                    f"Use Historical Volatility ({hist_vol:.1%})", 
                    value=True,
                    help="Use calculated historical volatility from recent price data"
                )
                
                if use_hist_vol:
                    sigma = hist_vol
                    st.success(f"Using historical volatility: {sigma:.1%}")
                else:
                    sigma = st.slider("Custom Volatility (%)", 5.0, 100.0, hist_vol*100, 2.5,
                                    help="Override with custom volatility estimate") / 100
            else:
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
            moneyness_pct = st.slider("Moneyness (%)", 25, 175, 100, 1,
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
                 help=f"Theoretical {option_type} option value calculated under the Black-Scholes framework")
    
    with metric_cols[1]:
        st.metric("Intrinsic Value", f"${intrinsic:.3f}",
                 help="Value if exercised immediately")
    
    with metric_cols[2]:
        st.metric("Time Value", f"${time_value:.3f}",
                 help="Premium above intrinsic value")
    
    with metric_cols[3]:
        leverage = (delta * S) / price if price > 0.001 else 0
        st.metric("Leverage", f"{leverage:.1f}x",
                 help="Effective leverage of the option vs. owning the underlying stock")
        
    with metric_cols[4]:
        breakeven = K + price if option_type == 'call' else K - price
        st.metric("Breakeven", f"${breakeven:.2f}",
                 help="Stock price needed to break even")
    
    # === ENHANCED GREEKS DASHBOARD ===
    st.subheader("📊 Greeks Analytics")
    
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
    st.subheader("📈 Advanced Visualizations")
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "📊 Greeks Surface", "💹 P&L Analysis", "⏰ Time Decay", "🌊 Volatility Impact", "🎲 3D Price Surface"
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
        pnl_col1.metric("Max Profit", f"${max(option_pnl):.2f}" if max(option_pnl) < (price+1) else "Unlimited")
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
    
    with viz_tab5:
        st.write("**3D Option Price Surface (Stock Price × Time)**")
        
        st.markdown("""
        This 3D surface shows how the option value changes with both stock price movements and time decay.
        The surface demonstrates the **non-linear** relationship between these factors.
        """)
        
        # Create 3D mesh grid
        S_3d_min = S * 0.7
        S_3d_max = S * 1.3
        S_3d_range = np.linspace(S_3d_min, S_3d_max, 40)
        T_3d_range = np.linspace(T, max(T * 0.1, 1/365), 40)  # From current T down to near expiry
        
        S_mesh, T_mesh = np.meshgrid(S_3d_range, T_3d_range)
        price_mesh = np.zeros_like(S_mesh)
        
        # Calculate option prices across the surface
        with st.spinner("Generating 3D surface..."):
            for i in range(len(T_3d_range)):
                for j in range(len(S_3d_range)):
                    try:
                        price_mesh[i, j] = bs_price(
                            S_mesh[i, j], K, T_mesh[i, j], r, sigma, option_type
                        )
                    except:
                        price_mesh[i, j] = np.nan
        
        # Create 3D surface plot
        fig_3d = go.Figure(data=[go.Surface(
            x=S_mesh,
            y=T_mesh * 365,  # Convert to days for better readability
            z=price_mesh,
            colorscale='Viridis',
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
            },
            colorbar=dict(title="Option<br>Price ($)")
        )])
        
        # Add current position marker
        fig_3d.add_trace(go.Scatter3d(
            x=[S],
            y=[T * 365],
            z=[price],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Current Position',
            showlegend=True
        ))
        
        # Update layout
        fig_3d.update_layout(
            title=dict(
                text=f'Black-Scholes {option_type.title()} Option Price Surface<br><sub>Strike: ${K:.2f} | Vol: {sigma:.1%} | Rate: {r:.1%}</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='Stock Price ($)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                yaxis=dict(title='Time to Expiry (days)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                zaxis=dict(title='Option Price ($)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=80)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Add interactive controls
        col_3d1, col_3d2 = st.columns(2)
        
        with col_3d1:
            st.markdown("""
            **🎯 How to Read This Chart:**
            - **X-axis:** Stock price range (±30% from current)
            - **Y-axis:** Time remaining until expiration
            - **Z-axis (height):** Option value
            - **Red diamond:** Your current position
            """)
        
        with col_3d2:
            st.markdown("""
            **📊 Key Observations:**
            - **Surface steepness** = Delta sensitivity
            - **Curvature** = Gamma effects
            - **Downward slope (time axis)** = Theta decay
            - **Distance from intrinsic value** = Time premium
            """)
        
        # Cross-section analysis
        st.markdown("---")
        st.write("**📍 Cross-Section Analysis**")
        
        slice_option = st.radio(
            "Select cross-section view:",
            ["Fixed Time (vary price)", "Fixed Price (vary time)"],
            horizontal=True
        )
        
        if slice_option == "Fixed Time (vary price)":
            # Show how price changes with stock price at current time
            fixed_T = T
            slice_prices = [bs_price(s, K, fixed_T, r, sigma, option_type) for s in S_3d_range]
            
            fig_slice = go.Figure()
            fig_slice.add_trace(go.Scatter(
                x=S_3d_range,
                y=slice_prices,
                mode='lines',
                name=f'Option Price (T={T_days} days)',
                line=dict(color='blue', width=3)
            ))
            
            # Add intrinsic value line
            if option_type == 'call':
                intrinsic_line = [max(0, s - K) for s in S_3d_range]
            else:
                intrinsic_line = [max(0, K - s) for s in S_3d_range]
            
            fig_slice.add_trace(go.Scatter(
                x=S_3d_range,
                y=intrinsic_line,
                mode='lines',
                name='Intrinsic Value',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_slice.add_vline(x=S, line_dash="dot", line_color="black", annotation_text="Current")
            fig_slice.add_vline(x=K, line_dash="dot", line_color="red", annotation_text="Strike")
            
            fig_slice.update_layout(
                title=f"Price vs Stock Price (T = {T_days} days)",
                xaxis_title="Stock Price ($)",
                yaxis_title="Option Value ($)",
                height=400
            )
            
            st.plotly_chart(fig_slice, use_container_width=True)
            
        else:
            # Show how price changes with time at current stock price
            fixed_S = S
            time_days = T_3d_range * 365
            slice_prices = [bs_price(fixed_S, K, t, r, sigma, option_type) for t in T_3d_range]
            
            fig_slice = go.Figure()
            fig_slice.add_trace(go.Scatter(
                x=time_days,
                y=slice_prices,
                mode='lines',
                name=f'Option Price (S=${fixed_S:.2f})',
                line=dict(color='purple', width=3)
            ))
            
            # Add intrinsic value line
            if option_type == 'call':
                intrinsic_val = max(0, fixed_S - K)
            else:
                intrinsic_val = max(0, K - fixed_S)
            
            fig_slice.add_hline(y=intrinsic_val, line_dash="dash", line_color="red",
                              annotation_text="Intrinsic Value")
            fig_slice.add_vline(x=T_days, line_dash="dot", line_color="black",
                              annotation_text="Current")
            
            fig_slice.update_layout(
                title=f"Price vs Time to Expiry (S = ${fixed_S:.2f})",
                xaxis_title="Time to Expiry (days)",
                yaxis_title="Option Value ($)",
                height=400,
                xaxis=dict(autorange="reversed")  # Time goes down as we approach expiry
            )
            
            st.plotly_chart(fig_slice, use_container_width=True)
    
    # === MARKET DATA INTEGRATION ===
    st.subheader("📊 Market Data Integration")
    
    use_live_data = st.checkbox(
        "Use Live Market Option Prices", 
        help="Fetch real option prices from the market instead of manual input"
    )
    
    market_option_data = None
    market_price = 0.0
    
    if use_live_data:
        st.info("""
        **Live Market Data Features:**
        - Real-time bid/ask/mid prices from Yahoo Finance
        - Market implied volatility 
        - Volume and open interest data
        - Multiple expiration dates available
        - Automatic strike matching (closest available)
        
        **Limitations:**
        - US equities only • 15-20 min delay • Limited to liquid options
        """)
        
        try:
            from market_options import get_market_option_price
            import yfinance as yf
            
            # Get available expiration dates
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            
            if exp_dates:
                # Select expiration date closest to our T parameter
                target_days = T_days
                exp_date_options = []
                
                for exp_str in exp_dates[:6]:  # Limit to first 6 for performance
                    exp_dt = pd.to_datetime(exp_str)
                    days_diff = (exp_dt - pd.Timestamp.now()).days
                    exp_date_options.append((exp_str, days_diff))
                
                # Find closest expiration to our target
                closest_exp = min(exp_date_options, key=lambda x: abs(x[1] - target_days))
                selected_exp = closest_exp[0]
                actual_days = closest_exp[1]
                
                st.write(f"**Selected Expiration:** {selected_exp} ({actual_days} days)")
                
                # Fetch market data for this option
                with st.spinner("Fetching live market data..."):
                    market_option_data = get_market_option_price(ticker, K, selected_exp, option_type)
                
                if market_option_data:
                    # Display market data
                    market_col1, market_col2, market_col3, market_col4 = st.columns(4)
                    
                    market_col1.metric("Bid", f"${market_option_data['bid']:.3f}")
                    market_col2.metric("Ask", f"${market_option_data['ask']:.3f}")
                    market_col3.metric("Mid Price", f"${market_option_data['mid_price']:.3f}")
                    market_col4.metric("Last Price", f"${market_option_data['last_price']:.3f}")
                    
                    # Additional market info
                    info_col1, info_col2, info_col3 = st.columns(3)
                    info_col1.metric("Volume", f"{market_option_data['volume']:,.0f}")
                    info_col2.metric("Open Interest", f"{market_option_data['open_interest']:,.0f}")
                    info_col3.metric("Bid-Ask Spread", f"{market_option_data['spread_pct']:.1f}%")
                    
                    # Use mid price as the market price
                    market_price = market_option_data['mid_price']
                    
                    # Compare with theoretical price
                    theory_vs_market = price - market_price
                    
                    if abs(theory_vs_market) > 0.05:
                        if theory_vs_market > 0:
                            st.warning(f"⚠️ Theoretical price ${theory_vs_market:.3f} higher than market - option may be undervalued")
                        else:
                            st.info(f"ℹ️ Market price ${abs(theory_vs_market):.3f} higher than theoretical - option may be overvalued")
                    else:
                        st.success("✅ Theoretical and market prices are closely aligned")
                        
                    # Update T to match actual expiration
                    T = actual_days / 365
                    T_days = actual_days
                    
                    # Recalculate with actual time to expiration
                    price = bs_price(S, K, T, r, sigma, option=option_type)
                    delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option=option_type)
                    
                else:
                    st.warning("Could not fetch market data for this option. Using manual input.")
                    use_live_data = False
                    
            else:
                st.warning("No options data available for this ticker. Using manual input.")
                use_live_data = False
                
        except ImportError:
            st.warning("Market data module not available. Please ensure market_options.py is in the same directory.")
            use_live_data = False
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            use_live_data = False
    
# === IMPLIED VOLATILITY SECTION ===
st.subheader("🔍 Advanced Implied Volatility Analysis")

# Add option to fetch live IV data
iv_data_mode = st.radio(
    "IV Data Source",
    ["Manual Entry", "Live Market Data (IV Surface)"],
    horizontal=True,
    help="Choose to manually enter a price or fetch live options chain data"
)

if iv_data_mode == "Live Market Data (IV Surface)":
    st.info("📡 Fetching live options chain to calculate implied volatility surface and time series")
    
    try:
        import yfinance as yf
        
        # Get available expiration dates
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        
        if not exp_dates or len(exp_dates) == 0:
            st.warning("No options data available for this ticker. Please use Manual Entry mode.")
        else:
            # Let user select expiration dates
            iv_exp_col1, iv_exp_col2 = st.columns(2)
            
            with iv_exp_col1:
                # Show available expirations with days to expiry
                exp_options = []
                for exp_str in exp_dates[:12]:  # Show first 12 expirations
                    exp_dt = pd.to_datetime(exp_str)
                    days_to_exp = (exp_dt - pd.Timestamp.now()).days
                    exp_options.append(f"{exp_str} ({days_to_exp}d)")
                
                selected_exp_display = st.selectbox(
                    "Select Expiration Date",
                    exp_options,
                    help="Choose expiration for IV analysis"
                )
                
                # Extract actual date from display string
                selected_exp = selected_exp_display.split(" (")[0]
                selected_exp_dt = pd.to_datetime(selected_exp)
                days_to_selected_exp = (selected_exp_dt - pd.Timestamp.now()).days
                T_selected = days_to_selected_exp / 365
            
            with iv_exp_col2:
                # Option type for IV analysis
                iv_option_type = st.selectbox(
                    "Option Type for IV",
                    ["call", "put"],
                    index=0 if option_type == "call" else 1
                )
            
            # Fetch options chain
            with st.spinner(f"Fetching {iv_option_type} options chain for {selected_exp}..."):
                try:
                    opt_chain = stock.option_chain(selected_exp)
                    
                    if iv_option_type == "call":
                        chain_df = opt_chain.calls
                    else:
                        chain_df = opt_chain.puts
                    
                    # Filter out options with zero bid/ask
                    chain_df = chain_df[
                        (chain_df['bid'] > 0) & 
                        (chain_df['ask'] > 0) &
                        (chain_df['volume'] > 0)
                    ].copy()
                    
                    if len(chain_df) == 0:
                        st.warning("No liquid options found for this expiration.")
                    else:
                        # Calculate IV for each strike
                        chain_df['mid_price'] = (chain_df['bid'] + chain_df['ask']) / 2
                        chain_df['implied_vol'] = np.nan
                        
                        for idx, row in chain_df.iterrows():
                            try:
                                def option_price_diff(vol):
                                    return bs_price(S, row['strike'], T_selected, r, vol, iv_option_type) - row['mid_price']
                                
                                iv = brentq(option_price_diff, 0.001, 5.0)
                                chain_df.at[idx, 'implied_vol'] = iv
                            except:
                                chain_df.at[idx, 'implied_vol'] = np.nan
                        
                        # Remove failed calculations
                        chain_df = chain_df.dropna(subset=['implied_vol'])
                        chain_df = chain_df[chain_df['implied_vol'] > 0]
                        
                        if len(chain_df) > 0:
                            # Calculate moneyness
                            chain_df['moneyness'] = chain_df['strike'] / S
                            
                            st.success(f"✅ Calculated IV for {len(chain_df)} strikes")
                            
                            # Display IV metrics
                            iv_metrics_col1, iv_metrics_col2, iv_metrics_col3, iv_metrics_col4 = st.columns(4)
                            
                            iv_metrics_col1.metric("Min IV", f"{chain_df['implied_vol'].min():.1%}")
                            iv_metrics_col2.metric("Max IV", f"{chain_df['implied_vol'].max():.1%}")
                            iv_metrics_col3.metric("Mean IV", f"{chain_df['implied_vol'].mean():.1%}")
                            iv_metrics_col4.metric("ATM IV", f"{chain_df.iloc[(chain_df['moneyness'] - 1).abs().argsort()[:1]]['implied_vol'].values[0]:.1%}")
                            
                            # === IV SMILE/SKEW CHART ===
                            st.markdown("---")
                            st.write("**📈 Implied Volatility Smile/Skew**")
                            
                            fig_iv_smile = go.Figure()
                            
                            # Plot IV vs Strike
                            fig_iv_smile.add_trace(go.Scatter(
                                x=chain_df['strike'],
                                y=chain_df['implied_vol'] * 100,
                                mode='markers+lines',
                                name='Implied Volatility',
                                marker=dict(size=8, color=chain_df['volume'], colorscale='Viridis', 
                                          showscale=True, colorbar=dict(title="Volume")),
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add current stock price line
                            fig_iv_smile.add_vline(x=S, line_dash="dash", line_color="red",
                                                  annotation_text="Current Price")
                            
                            # Add historical volatility line if available
                            if online_vol_available:
                                fig_iv_smile.add_hline(y=hist_vol * 100, line_dash="dot", 
                                                      line_color="green",
                                                      annotation_text="Historical Vol")
                            
                            fig_iv_smile.update_layout(
                                title=f"IV Smile/Skew - {iv_option_type.title()} Options ({selected_exp})",
                                xaxis_title="Strike Price ($)",
                                yaxis_title="Implied Volatility (%)",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_iv_smile, use_container_width=True)
                            
                            # === IV TIME SERIES (Multiple Expirations) ===
                            st.markdown("---")
                            st.write("**📊 IV Term Structure (Time Series)**")
                            
                            with st.spinner("Fetching IV across multiple expirations..."):
                                iv_term_data = []
                                
                                # Select ATM or user-specified moneyness
                                term_moneyness = st.slider(
                                    "Moneyness for Term Structure",
                                    0.8, 1.2, 1.0, 0.05,
                                    help="1.0 = ATM, <1.0 = OTM calls/ITM puts, >1.0 = ITM calls/OTM puts"
                                )
                                
                                target_strike = S * term_moneyness
                                
                                for exp_date in exp_dates[:8]:  # Limit to 8 expirations for performance
                                    try:
                                        exp_dt = pd.to_datetime(exp_date)
                                        days_exp = (exp_dt - pd.Timestamp.now()).days
                                        T_exp = days_exp / 365
                                        
                                        opt_chain_exp = stock.option_chain(exp_date)
                                        chain_exp = opt_chain_exp.calls if iv_option_type == "call" else opt_chain_exp.puts
                                        
                                        # Find closest strike to target
                                        chain_exp = chain_exp[
                                            (chain_exp['bid'] > 0) & 
                                            (chain_exp['ask'] > 0)
                                        ].copy()
                                        
                                        if len(chain_exp) > 0:
                                            closest_idx = (chain_exp['strike'] - target_strike).abs().idxmin()
                                            closest_row = chain_exp.loc[closest_idx]
                                            
                                            mid_price = (closest_row['bid'] + closest_row['ask']) / 2
                                            
                                            try:
                                                def price_diff(vol):
                                                    return bs_price(S, closest_row['strike'], T_exp, r, vol, iv_option_type) - mid_price
                                                
                                                iv_val = brentq(price_diff, 0.001, 5.0)
                                                
                                                iv_term_data.append({
                                                    'expiration': exp_date,
                                                    'days_to_expiry': days_exp,
                                                    'strike': closest_row['strike'],
                                                    'iv': iv_val,
                                                    'volume': closest_row['volume'],
                                                    'open_interest': closest_row['openInterest']
                                                })
                                            except:
                                                pass
                                    except:
                                        pass
                                
                                if len(iv_term_data) > 0:
                                    iv_term_df = pd.DataFrame(iv_term_data)
                                    
                                    # Plot IV term structure
                                    fig_iv_term = go.Figure()
                                    
                                    fig_iv_term.add_trace(go.Scatter(
                                        x=iv_term_df['days_to_expiry'],
                                        y=iv_term_df['iv'] * 100,
                                        mode='markers+lines',
                                        name='Implied Volatility',
                                        marker=dict(size=10, color='blue'),
                                        line=dict(color='blue', width=3)
                                    ))
                                    
                                    # Add HV line if available
                                    if online_vol_available:
                                        fig_iv_term.add_hline(
                                            y=hist_vol * 100,
                                            line_dash="dash",
                                            line_color="green",
                                            annotation_text="Historical Volatility"
                                        )
                                    
                                    fig_iv_term.update_layout(
                                        title=f"IV Term Structure - {iv_option_type.title()} Options (Moneyness: {term_moneyness:.0%})",
                                        xaxis_title="Days to Expiration",
                                        yaxis_title="Implied Volatility (%)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_iv_term, use_container_width=True)
                                    
                                    # Display term structure table
                                    with st.expander("📋 View IV Term Structure Data"):
                                        display_df = iv_term_df.copy()
                                        display_df['iv'] = display_df['iv'].apply(lambda x: f"{x:.2%}")
                                        display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
                                        st.dataframe(display_df, use_container_width=True)
                                else:
                                    st.warning("Could not calculate IV term structure. Try different moneyness level.")
                            
                            # === 3D IV SURFACE ===
                            st.markdown("---")
                            st.write("**🎲 3D Implied Volatility Surface**")
                            
                            with st.spinner("Building 3D IV surface..."):
                                # Collect IV data across strikes and expirations
                                surface_data = []
                                
                                for exp_date in exp_dates[:6]:  # Limit to 6 expirations
                                    try:
                                        exp_dt = pd.to_datetime(exp_date)
                                        days_exp = (exp_dt - pd.Timestamp.now()).days
                                        T_exp = days_exp / 365
                                        
                                        opt_chain_exp = stock.option_chain(exp_date)
                                        chain_exp = opt_chain_exp.calls if iv_option_type == "call" else opt_chain_exp.puts
                                        
                                        chain_exp = chain_exp[
                                            (chain_exp['bid'] > 0) & 
                                            (chain_exp['ask'] > 0) &
                                            (chain_exp['volume'] > 0)
                                        ].copy()
                                        
                                        for idx, row in chain_exp.iterrows():
                                            try:
                                                mid_price = (row['bid'] + row['ask']) / 2
                                                
                                                def price_diff(vol):
                                                    return bs_price(S, row['strike'], T_exp, r, vol, iv_option_type) - mid_price
                                                
                                                iv_val = brentq(price_diff, 0.001, 5.0)
                                                
                                                surface_data.append({
                                                    'strike': row['strike'],
                                                    'days_to_expiry': days_exp,
                                                    'moneyness': row['strike'] / S,
                                                    'iv': iv_val
                                                })
                                            except:
                                                pass
                                    except:
                                        pass
                                
                                if len(surface_data) > 10:
                                    surface_df = pd.DataFrame(surface_data)
                                    
                                    # Create pivot table for surface
                                    pivot_strikes = sorted(surface_df['strike'].unique())
                                    pivot_days = sorted(surface_df['days_to_expiry'].unique())
                                    
                                    # Create mesh grid
                                    Z_iv = []
                                    for day in pivot_days:
                                        row_data = []
                                        for strike in pivot_strikes:
                                            matching = surface_df[
                                                (surface_df['strike'] == strike) & 
                                                (surface_df['days_to_expiry'] == day)
                                            ]
                                            if len(matching) > 0:
                                                row_data.append(matching.iloc[0]['iv'] * 100)
                                            else:
                                                row_data.append(np.nan)
                                        Z_iv.append(row_data)
                                    
                                    Z_iv = np.array(Z_iv)
                                    X_strikes, Y_days = np.meshgrid(pivot_strikes, pivot_days)
                                    
                                    # Create 3D surface
                                    fig_iv_surface = go.Figure(data=[go.Surface(
                                        x=X_strikes,
                                        y=Y_days,
                                        z=Z_iv,
                                        colorscale='Plasma',
                                        colorbar=dict(title="IV (%)"),
                                        contours={
                                            "z": {"show": True, "usecolormap": True, 
                                                 "highlightcolor": "limegreen", "project": {"z": True}}
                                        }
                                    )])
                                    
                                    # Add current position marker if applicable
                                    if T_days in pivot_days or abs(T_days - min(pivot_days, key=lambda x: abs(x - T_days))) < 5:
                                        closest_day = min(pivot_days, key=lambda x: abs(x - T_days))
                                        closest_strike = min(pivot_strikes, key=lambda x: abs(x - K))
                                        
                                        # Find IV at this point
                                        matching_iv = surface_df[
                                            (abs(surface_df['strike'] - closest_strike) < 0.5) &
                                            (abs(surface_df['days_to_expiry'] - closest_day) < 2)
                                        ]
                                        
                                        if len(matching_iv) > 0:
                                            fig_iv_surface.add_trace(go.Scatter3d(
                                                x=[closest_strike],
                                                y=[closest_day],
                                                z=[matching_iv.iloc[0]['iv'] * 100],
                                                mode='markers',
                                                marker=dict(size=8, color='red', symbol='diamond'),
                                                name='Your Position',
                                                showlegend=True
                                            ))
                                    
                                    fig_iv_surface.update_layout(
                                        title=f"Implied Volatility Surface - {iv_option_type.title()} Options",
                                        scene=dict(
                                            xaxis=dict(title='Strike Price ($)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                                            yaxis=dict(title='Days to Expiry', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                                            zaxis=dict(title='Implied Volatility (%)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                                        ),
                                        height=700,
                                        margin=dict(l=0, r=0, b=0, t=80)
                                    )
                                    
                                    st.plotly_chart(fig_iv_surface, use_container_width=True)
                                    
                                    # IV Surface insights
                                    insights_col1, insights_col2 = st.columns(2)
                                    
                                    with insights_col1:
                                        st.markdown("""
                                        **🔍 IV Surface Interpretation:**
                                        - **Volatility Smile:** IV typically higher for OTM options (wings)
                                        - **Volatility Skew:** Asymmetry between put and call IV
                                        - **Surface shows market's expected price distribution**
                                        """)
                                    
                                    with insights_col2:
                                        st.markdown("""
                                        **📊 Trading Insights:**
                                        - **High IV areas:** Expensive options, potential credit strategies
                                        - **Low IV areas:** Cheap options, potential debit strategies
                                        - **IV vs HV:** Compare to historical volatility for relative value
                                        """)
                                    
                                    # Add IV vs HV comparison if available
                                    if online_vol_available:
                                        st.markdown("---")
                                        st.write("**📉 IV vs Historical Volatility Comparison**")
                                        
                                        # Calculate average IV across the surface
                                        avg_iv = surface_df['iv'].mean()
                                        
                                        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
                                        comparison_col1.metric("Average IV", f"{avg_iv:.2%}")
                                        comparison_col2.metric("Historical Vol (20d)", f"{hist_vol:.2%}")
                                        
                                        iv_hv_ratio = avg_iv / hist_vol
                                        comparison_col3.metric("IV/HV Ratio", f"{iv_hv_ratio:.2f}x")
                                        
                                        # Interpretation
                                        if iv_hv_ratio > 1.2:
                                            st.warning("🔺 **High IV Environment:** Options are expensive relative to historical volatility. Consider credit strategies or selling premium.")
                                        elif iv_hv_ratio < 0.8:
                                            st.info("🔻 **Low IV Environment:** Options are cheap relative to historical volatility. Consider debit strategies or buying premium.")
                                        else:
                                            st.success("⚖️ **Balanced Environment:** IV is in line with historical volatility. Fair pricing indicated.")
                                        
                                        # Create IV vs HV time series comparison
                                        fig_iv_hv = go.Figure()
                                        
                                        fig_iv_hv.add_trace(go.Scatter(
                                            x=iv_term_df['days_to_expiry'] if len(iv_term_data) > 0 else [],
                                            y=iv_term_df['iv'] * 100 if len(iv_term_data) > 0 else [],
                                            mode='lines+markers',
                                            name='Implied Volatility',
                                            line=dict(color='blue', width=3),
                                            marker=dict(size=8)
                                        ))
                                        
                                        fig_iv_hv.add_hline(
                                            y=hist_vol * 100,
                                            line_dash="dash",
                                            line_color="green",
                                            line_width=3,
                                            annotation_text="Historical Volatility (20d)"
                                        )
                                        
                                        fig_iv_hv.update_layout(
                                            title="IV vs HV: Term Structure Comparison",
                                            xaxis_title="Days to Expiration",
                                            yaxis_title="Volatility (%)",
                                            height=400,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig_iv_hv, use_container_width=True)
                                else:
                                    st.warning("Insufficient data to build 3D IV surface. Need more liquid strikes across expirations.")
                        else:
                            st.warning("Could not calculate IV - no valid option prices found.")
                
                except Exception as e:
                    st.error(f"Error fetching options chain: {str(e)}")
    
    except ImportError:
        st.error("yfinance library required for live market data")
    except Exception as e:
        st.error(f"Error in live data mode: {str(e)}")

else:
    # Manual Entry Mode (Original code)
    st.markdown("**Manual IV Calculator**")
    
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
                if 'hist_vol' in locals() and online_vol_available:
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
    options_page()
