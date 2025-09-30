import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils import bs_price, greeks, d1, d2

def iv_analysis_page():
    st.title("🔍 Implied Volatility Analysis & Surface Visualization")
    
    # Add descriptive banner
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
    <strong>Advanced IV Analytics:</strong> Analyze implied volatility surfaces, term structures, and volatility smiles using live market data. 
    Compare IV vs HV to identify trading opportunities and understand market sentiment.
    </div>
    """, unsafe_allow_html=True)
    
    # === STOCK SELECTION ===
    st.subheader("📊 Select Underlying Asset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Popular tickers for options trading
        popular_tickers = {
            "SPY - S&P 500 ETF": "SPY",
            "QQQ - Nasdaq 100 ETF": "QQQ",
            "IWM - Russell 2000 ETF": "IWM",
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Tesla": "TSLA",
            "Amazon": "AMZN",
            "Nvidia": "NVDA",
            "Meta": "META",
            "Google": "GOOGL"
        }
        
        ticker_choice = st.selectbox(
            "Select from popular tickers or enter custom below",
            [""] + list(popular_tickers.keys()),
            help="Choose highly liquid options for best results"
        )
        
        if ticker_choice:
            ticker = popular_tickers[ticker_choice]
            st.success(f"Selected: {ticker}")
        else:
            ticker = st.text_input("Enter Ticker Symbol", "SPY", help="Use highly liquid tickers for best results")
    
    with col2:
        # Risk-free rate input
        r = st.slider("Risk-free Rate (%)", 0.0, 15.0, 5.0, 0.25) / 100
        st.info(f"Using rate: {r:.2%}")
    
    # Fetch stock data
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="60d")
        
        if data.empty:
            st.error("❌ Invalid ticker or no data available. Please try another ticker.")
            return
        
        S = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2] if len(data) > 1 else S
        change_pct = (S - prev_close) / prev_close * 100
        
        # Calculate historical volatility
        if len(data) >= 20:
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            hist_vol_20d = returns.tail(20).std() * np.sqrt(252)
            hist_vol_60d = returns.std() * np.sqrt(252)
            online_vol_available = True
        else:
            hist_vol_20d = 0.2
            hist_vol_60d = 0.2
            online_vol_available = False
        
        # Display stock info
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        info_col1.metric("Current Price", f"${S:.2f}", f"{change_pct:+.2f}%")
        info_col2.metric("HV (20d)", f"{hist_vol_20d:.1%}" if online_vol_available else "N/A")
        info_col3.metric("HV (60d)", f"{hist_vol_60d:.1%}" if online_vol_available else "N/A")
        
        # Get company name if available
        try:
            info = stock.info
            company_name = info.get('longName', ticker)
            info_col4.metric("Company", company_name if len(company_name) < 15 else ticker)
        except:
            info_col4.metric("Ticker", ticker)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return
    
    st.markdown("---")
    
    # === IMPLIED VOLATILITY ANALYSIS SECTION ===
    st.subheader("🎲 Implied Volatility Surface & Analysis")
    
    # Mode selection
    iv_data_mode = st.radio(
        "Analysis Mode",
        ["Live Market Data (Full IV Surface)", "Manual IV Calculator"],
        horizontal=True,
        help="Choose live data for comprehensive analysis or manual for single option IV calculation"
    )
    
    if iv_data_mode == "Live Market Data (Full IV Surface)":
        st.info("📡 Fetching live options chain to calculate implied volatility surface and term structure")
        
        # Get available expiration dates
        exp_dates = stock.options
        
        if not exp_dates or len(exp_dates) == 0:
            st.warning("⚠️ No options data available for this ticker. Please try another ticker or use Manual mode.")
            return
        
        # User selections
        iv_exp_col1, iv_exp_col2 = st.columns(2)
        
        with iv_exp_col1:
            # Show available expirations with days to expiry
            exp_options = []
            for exp_str in exp_dates[:12]:
                exp_dt = pd.to_datetime(exp_str)
                days_to_exp = (exp_dt - pd.Timestamp.now()).days
                exp_options.append(f"{exp_str} ({days_to_exp}d)")
            
            selected_exp_display = st.selectbox(
                "Primary Expiration Date",
                exp_options,
                index=min(2, len(exp_options)-1),  # Default to ~3rd expiration
                help="Select expiration for detailed IV smile analysis"
            )
            
            selected_exp = selected_exp_display.split(" (")[0]
            selected_exp_dt = pd.to_datetime(selected_exp)
            days_to_selected_exp = (selected_exp_dt - pd.Timestamp.now()).days
            T_selected = days_to_selected_exp / 365
        
        with iv_exp_col2:
            iv_option_type = st.selectbox(
                "Option Type",
                ["call", "put"],
                help="Select call or put options for analysis"
            )
        
        # Fetch and analyze options chain
        with st.spinner(f"📊 Fetching {iv_option_type} options chain for {selected_exp}..."):
            try:
                opt_chain = stock.option_chain(selected_exp)
                chain_df = opt_chain.calls if iv_option_type == "call" else opt_chain.puts
                
                # Filter liquid options
                chain_df = chain_df[
                    (chain_df['bid'] > 0) & 
                    (chain_df['ask'] > 0) &
                    (chain_df['volume'] > 0)
                ].copy()
                
                if len(chain_df) == 0:
                    st.warning("❌ No liquid options found for this expiration. Try another date.")
                    return
                
                # Calculate IV for each strike
                chain_df['mid_price'] = (chain_df['bid'] + chain_df['ask']) / 2
                chain_df['implied_vol'] = np.nan
                
                progress_bar = st.progress(0)
                for i, (idx, row) in enumerate(chain_df.iterrows()):
                    try:
                        def option_price_diff(vol):
                            return bs_price(S, row['strike'], T_selected, r, vol, iv_option_type) - row['mid_price']
                        
                        iv = brentq(option_price_diff, 0.001, 5.0)
                        chain_df.at[idx, 'implied_vol'] = iv
                    except:
                        chain_df.at[idx, 'implied_vol'] = np.nan
                    
                    progress_bar.progress((i + 1) / len(chain_df))
                
                progress_bar.empty()
                
                # Clean data
                chain_df = chain_df.dropna(subset=['implied_vol'])
                chain_df = chain_df[(chain_df['implied_vol'] > 0) & (chain_df['implied_vol'] < 3)]
                
                if len(chain_df) == 0:
                    st.warning("Could not calculate valid IVs. Try another expiration.")
                    return
                
                # Calculate moneyness
                chain_df['moneyness'] = chain_df['strike'] / S
                
                st.success(f"✅ Successfully calculated IV for {len(chain_df)} strikes")
                
                # Display IV summary metrics
                st.markdown("### 📈 IV Summary Statistics")
                iv_metrics_col1, iv_metrics_col2, iv_metrics_col3, iv_metrics_col4 = st.columns(4)
                
                atm_iv = chain_df.iloc[(chain_df['moneyness'] - 1).abs().argsort()[:1]]['implied_vol'].values[0]
                
                iv_metrics_col1.metric("Min IV", f"{chain_df['implied_vol'].min():.1%}")
                iv_metrics_col2.metric("ATM IV", f"{atm_iv:.1%}")
                iv_metrics_col3.metric("Max IV", f"{chain_df['implied_vol'].max():.1%}")
                iv_metrics_col4.metric("Mean IV", f"{chain_df['implied_vol'].mean():.1%}")
                
                # === IV SMILE/SKEW CHART ===
                st.markdown("---")
                st.markdown("### 📊 Implied Volatility Smile/Skew")
                
                fig_iv_smile = go.Figure()
                
                # Plot IV vs Strike
                fig_iv_smile.add_trace(go.Scatter(
                    x=chain_df['strike'],
                    y=chain_df['implied_vol'] * 100,
                    mode='markers+lines',
                    name='Implied Volatility',
                    marker=dict(
                        size=10, 
                        color=chain_df['volume'], 
                        colorscale='Viridis',
                        showscale=True, 
                        colorbar=dict(title="Volume", x=1.15)
                    ),
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Strike:</b> $%{x:.2f}<br><b>IV:</b> %{y:.2f}%<extra></extra>'
                ))
                
                # Add current stock price line
                fig_iv_smile.add_vline(
                    x=S, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Spot: ${S:.2f}",
                    annotation_position="top"
                )
                
                # Add historical volatility line if available
                if online_vol_available:
                    fig_iv_smile.add_hline(
                        y=hist_vol_20d * 100, 
                        line_dash="dot", 
                        line_color="green",
                        annotation_text=f"HV (20d): {hist_vol_20d:.1%}",
                        annotation_position="right"
                    )
                
                fig_iv_smile.update_layout(
                    title=f"IV Smile/Skew - {iv_option_type.title()} Options ({selected_exp})",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Implied Volatility (%)",
                    height=500,
                    hovermode='closest',
                    showlegend=False
                )
                
                st.plotly_chart(fig_iv_smile, use_container_width=True)
                
                # Skew analysis
                otm_strikes = chain_df[chain_df['moneyness'] < 0.95] if iv_option_type == "call" else chain_df[chain_df['moneyness'] > 1.05]
                itm_strikes = chain_df[chain_df['moneyness'] > 1.05] if iv_option_type == "call" else chain_df[chain_df['moneyness'] < 0.95]
                
                if len(otm_strikes) > 0 and len(itm_strikes) > 0:
                    avg_otm_iv = otm_strikes['implied_vol'].mean()
                    avg_itm_iv = itm_strikes['implied_vol'].mean()
                    skew = avg_otm_iv - avg_itm_iv
                    
                    skew_col1, skew_col2, skew_col3 = st.columns(3)
                    skew_col1.metric("OTM Avg IV", f"{avg_otm_iv:.2%}")
                    skew_col2.metric("ITM Avg IV", f"{avg_itm_iv:.2%}")
                    skew_col3.metric("Skew", f"{skew:.2%}", 
                                   help="Positive = OTM more expensive, Negative = ITM more expensive")
                
                # === IV TERM STRUCTURE ===
                st.markdown("---")
                st.markdown("### 📉 IV Term Structure (Time Series)")
                
                term_moneyness = st.slider(
                    "Moneyness Level for Term Structure",
                    0.80, 1.20, 1.00, 0.05,
                    help="1.0 = ATM, <1.0 = OTM calls/ITM puts, >1.0 = ITM calls/OTM puts",
                    format="%.2f"
                )
                
                with st.spinner("Calculating IV across multiple expirations..."):
                    iv_term_data = []
                    target_strike = S * term_moneyness
                    
                    for exp_date in exp_dates[:10]:
                        try:
                            exp_dt = pd.to_datetime(exp_date)
                            days_exp = (exp_dt - pd.Timestamp.now()).days
                            
                            if days_exp < 1:
                                continue
                            
                            T_exp = days_exp / 365
                            
                            opt_chain_exp = stock.option_chain(exp_date)
                            chain_exp = opt_chain_exp.calls if iv_option_type == "call" else opt_chain_exp.puts
                            
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
                    
                    if len(iv_term_data) >= 2:
                        iv_term_df = pd.DataFrame(iv_term_data)
                        
                        # Plot IV term structure
                        fig_iv_term = go.Figure()
                        
                        fig_iv_term.add_trace(go.Scatter(
                            x=iv_term_df['days_to_expiry'],
                            y=iv_term_df['iv'] * 100,
                            mode='markers+lines',
                            name='Implied Volatility',
                            marker=dict(size=12, color='blue', symbol='circle'),
                            line=dict(color='blue', width=3),
                            hovertemplate='<b>Days:</b> %{x}<br><b>IV:</b> %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Add HV reference lines
                        if online_vol_available:
                            fig_iv_term.add_hline(
                                y=hist_vol_20d * 100,
                                line_dash="dash",
                                line_color="green",
                                line_width=2,
                                annotation_text="HV (20d)"
                            )
                            
                            fig_iv_term.add_hline(
                                y=hist_vol_60d * 100,
                                line_dash="dot",
                                line_color="orange",
                                line_width=2,
                                annotation_text="HV (60d)"
                            )
                        
                        fig_iv_term.update_layout(
                            title=f"IV Term Structure - {iv_option_type.title()} ({term_moneyness:.0%} Moneyness)",
                            xaxis_title="Days to Expiration",
                            yaxis_title="Implied Volatility (%)",
                            height=450,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_iv_term, use_container_width=True)
                        
                        # Term structure insights
                        front_iv = iv_term_df.iloc[0]['iv']
                        back_iv = iv_term_df.iloc[-1]['iv']
                        term_structure_slope = back_iv - front_iv
                        
                        ts_col1, ts_col2, ts_col3 = st.columns(3)
                        ts_col1.metric("Near-term IV", f"{front_iv:.2%}")
                        ts_col2.metric("Far-term IV", f"{back_iv:.2%}")
                        ts_col3.metric("Term Structure", 
                                     "Upward" if term_structure_slope > 0.02 else "Flat" if abs(term_structure_slope) <= 0.02 else "Inverted",
                                     f"{term_structure_slope:+.2%}")
                        
                        # Display term structure table
                        with st.expander("📋 View Detailed Term Structure Data"):
                            display_df = iv_term_df.copy()
                            display_df['iv'] = display_df['iv'].apply(lambda x: f"{x:.2%}")
                            display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
                            display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}")
                            display_df['open_interest'] = display_df['open_interest'].apply(lambda x: f"{x:,.0f}")
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Insufficient data for term structure. Try different moneyness level.")
                
                # === 3D IV SURFACE ===
                st.markdown("---")
                st.markdown("### 🎲 3D Implied Volatility Surface")
                
                with st.spinner("Building 3D IV surface (this may take a moment)..."):
                    surface_data = []
                    
                    num_expirations = st.slider("Number of expirations to include", 3, 10, 6,
                                               help="More expirations = better surface but slower")
                    
                    for exp_date in exp_dates[:num_expirations]:
                        try:
                            exp_dt = pd.to_datetime(exp_date)
                            days_exp = (exp_dt - pd.Timestamp.now()).days
                            
                            if days_exp < 1:
                                continue
                            
                            T_exp = days_exp / 365
                            
                            opt_chain_exp = stock.option_chain(exp_date)
                            chain_exp = opt_chain_exp.calls if iv_option_type == "call" else opt_chain_exp.puts
                            
                            chain_exp = chain_exp[
                                (chain_exp['bid'] > 0) & 
                                (chain_exp['ask'] > 0) &
                                (chain_exp['volume'] > 0)
                            ].copy()
                            
                            for idx, row in chain_exp.iterrows():
                                # Filter reasonable strikes
                                if row['strike'] < S * 0.5 or row['strike'] > S * 1.5:
                                    continue
                                
                                try:
                                    mid_price = (row['bid'] + row['ask']) / 2
                                    
                                    def price_diff(vol):
                                        return bs_price(S, row['strike'], T_exp, r, vol, iv_option_type) - mid_price
                                    
                                    iv_val = brentq(price_diff, 0.001, 5.0)
                                    
                                    if 0 < iv_val < 3:  # Reasonable IV range
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
                    
                    if len(surface_data) > 20:
                        surface_df = pd.DataFrame(surface_data)
                        
                        # Create mesh grid
                        pivot_strikes = sorted(surface_df['strike'].unique())
                        pivot_days = sorted(surface_df['days_to_expiry'].unique())
                        
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
                            colorbar=dict(title="IV (%)", x=1.1),
                            contours={
                                "z": {
                                    "show": True, 
                                    "usecolormap": True,
                                    "highlightcolor": "limegreen",
                                    "project": {"z": True}
                                }
                            },
                            hovertemplate='<b>Strike:</b> $%{x:.2f}<br><b>Days:</b> %{y:.0f}<br><b>IV:</b> %{z:.2f}%<extra></extra>'
                        )])
                        
                        # Add current spot price plane
                        fig_iv_surface.add_trace(go.Scatter3d(
                            x=[S, S],
                            y=[min(pivot_days), max(pivot_days)],
                            z=[0, 100],
                            mode='lines',
                            line=dict(color='red', width=4, dash='dash'),
                            name='Spot Price',
                            showlegend=True
                        ))
                        
                        fig_iv_surface.update_layout(
                            title=f"3D Implied Volatility Surface - {ticker} {iv_option_type.title()}s",
                            scene=dict(
                                xaxis=dict(
                                    title='Strike Price ($)',
                                    backgroundcolor="rgb(240, 240, 240)",
                                    gridcolor="white"
                                ),
                                yaxis=dict(
                                    title='Days to Expiry',
                                    backgroundcolor="rgb(240, 240, 240)",
                                    gridcolor="white"
                                ),
                                zaxis=dict(
                                    title='Implied Volatility (%)',
                                    backgroundcolor="rgb(240, 240, 240)",
                                    gridcolor="white"
                                ),
                                camera=dict(
                                    eye=dict(x=1.6, y=1.6, z=1.4)
                                )
                            ),
                            height=700,
                            margin=dict(l=0, r=0, b=0, t=60)
                        )
                        
                        st.plotly_chart(fig_iv_surface, use_container_width=True)
                        
                        # IV Surface insights
                        st.markdown("---")
                        st.markdown("### 💡 IV Surface Interpretation & Trading Insights")
                        
                        insights_col1, insights_col2 = st.columns(2)
                        
                        with insights_col1:
                            st.markdown("""
                            **🔍 What the Surface Shows:**
                            - **Volatility Smile:** Wings (OTM options) typically have higher IV
                            - **Volatility Skew:** Asymmetry between put and call IV (fear gauge)
                            - **Term Structure:** How IV evolves with time to expiration
                            - **Surface curvature:** Market's view of tail risk and price distribution
                            """)
                        
                        with insights_col2:
                            st.markdown("""
                            **📊 Trading Applications:**
                            - **High IV zones:** Consider selling premium (credit spreads)
                            - **Low IV zones:** Consider buying premium (debit spreads)
                            - **Steep skew:** Indicates elevated crash/rally concerns
                            - **Flat surface:** Suggests stable, low-volatility environment
                            """)
                        
                        # === IV vs HV COMPARISON ===
                        if online_vol_available:
                            st.markdown("---")
                            st.markdown("### 📊 IV vs Historical Volatility Analysis")
                            
                            avg_iv = surface_df['iv'].mean()
                            
                            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                            comp_col1.metric("Average IV", f"{avg_iv:.2%}")
                            comp_col2.metric("HV (20d)", f"{hist_vol_20d:.2%}")
                            comp_col3.metric("HV (60d)", f"{hist_vol_60d:.2%}")
                            
                            iv_hv_ratio = avg_iv / hist_vol_20d
                            comp_col4.metric("IV/HV Ratio", f"{iv_hv_ratio:.2f}x")
                            
                            # Interpretation
                            if iv_hv_ratio > 1.3:
                                st.error("🔺 **High IV Environment** - Options are expensive relative to historical volatility")
                                st.markdown("""
                                **Suggested Strategies:** Credit spreads, iron condors, covered calls, cash-secured puts
                                """)
                            elif iv_hv_ratio < 0.7:
                                st.success("🔻 **Low IV Environment** - Options are cheap relative to historical volatility")
                                st.markdown("""
                                **Suggested Strategies:** Long straddles/strangles, debit spreads, protective puts
                                """)
                            else:
                                st.info("⚖️ **Balanced Environment** - IV is fairly priced relative to historical volatility")
                                st.markdown("""
                                **Suggested Strategies:** Directional trades, calendar spreads, butterflies
                                """)
                            
                            # Create comparative visualization
                            if len(iv_term_data) > 0:
                                fig_comparison = go.Figure()
                                
                                fig_comparison.add_trace(go.Scatter(
                                    x=iv_term_df['days_to_expiry'],
                                    y=iv_term_df['iv'] * 100,
                                    mode='lines+markers',
                                    name='Implied Volatility',
                                    line=dict(color='blue', width=3),
                                    marker=dict(size=10)
                                ))
                                
                                fig_comparison.add_hline(
                                    y=hist_vol_20d * 100,
                                    line_dash="dash",
                                    line_color="green",
                                    line_width=3,
                                    annotation_text="Historical Vol (20d)",
                                    annotation_position="right"
                                )
                                
                                fig_comparison.add_hline(
                                    y=hist_vol_60d * 100,
                                    line_dash="dot",
                                    line_color="orange",
                                    line_width=3,
                                    annotation_text="Historical Vol (60d)",
                                    annotation_position="right"
                                )
                                
                                fig_comparison.update_layout(
                                    title="IV vs HV: Term Structure Comparison",
                                    xaxis_title="Days to Expiration",
                                    yaxis_title="Volatility (%)",
                                    height=450,
                                    showlegend=True,
                                    legend=dict(x=0.02, y=0.98)
                                )
                                
                                st.plotly_chart(fig_comparison, use_container_width=True)
                    else:
                        st.warning("⚠️ Insufficient data to build 3D IV surface. Try a more liquid ticker (e.g., SPY, QQQ, AAPL).")
                
            except Exception as e:
                st.error(f"Error processing options chain: {str(e)}")
                st.info("Try selecting a different expiration date or a more liquid ticker.")
    
    else:
        # === MANUAL IV CALCULATOR ===
        st.markdown("### 🔢 Manual IV Calculator")
        st.info("Enter option parameters to calculate implied volatility from market price")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.markdown("**Option Parameters**")
            
            option_type = st.selectbox("Option Type", ["call", "put"])
            K = st.number_input("Strike Price ($)", value=float(S), min_value=0.01, step=0.50)
            
            expiry_method = st.radio("Time Input Method", ["Days", "Date"], horizontal=True)
            
            if expiry_method == "Days":
                T_days = st.number_input("Days to Expiration", value=30, min_value=1, max_value=730)
                T = T_days / 365
            else:
                exp_date = st.date_input("Expiration Date", 
                                        value=pd.Timestamp.now() + pd.Timedelta(days=30))
                T_days = (exp_date - pd.Timestamp.now().date()).days
                T = max(T_days / 365, 0.001)
                st.info(f"Days to expiry: {T_days}")
            
            market_price = st.number_input(
                "Market Option Price ($)",
                value=5.0,
                min_value=0.01,
                step=0.01,
                help="Enter the observed market price of the option"
            )
        
        with calc_col2:
            st.markdown("**Calculation Results**")
            
            if market_price > 0:
                try:
                    # Calculate IV
                    def option_price_diff(vol):
                        return bs_price(S, K, T, r, vol, option_type) - market_price
                    
                    implied_vol = brentq(option_price_diff, 0.001, 5.0)
                    
                    st.success(f"### {implied_vol:.2%}")
                    st.caption("Implied Volatility")
                    
                    # Calculate theoretical price at calculated IV
                    theo_price = bs_price(S, K, T, r, implied_vol, option_type)
                    
                    st.metric("Theoretical Price", f"${theo_price:.3f}")
                    st.metric("Market Price", f"${market_price:.3f}")
                    
                    # Moneyness indicator
                    moneyness = S / K
                    if option_type == "call":
                        if moneyness > 1.05:
                            status = "🟢 In-The-Money"
                        elif moneyness > 0.95:
                            status = "🔵 At-The-Money"
                        else:
                            status = "🔴 Out-Of-The-Money"
                    else:
                        if moneyness < 0.95:
                            status = "🟢 In-The-Money"
                        elif moneyness < 1.05:
                            status = "🔵 At-The-Money"
                        else:
                            status = "🔴 Out-Of-The-Money"
                    
                    st.info(f"**Status:** {status}")
                    
                    # Compare with HV if available
                    if online_vol_available:
                        st.markdown("---")
                        st.markdown("**IV vs Historical Volatility**")
                        
                        iv_hv_ratio_20d = implied_vol / hist_vol_20d
                        iv_hv_ratio_60d = implied_vol / hist_vol_60d
                        
                        hv_col1, hv_col2 = st.columns(2)
                        hv_col1.metric("HV (20d)", f"{hist_vol_20d:.2%}")
                        hv_col2.metric("IV/HV Ratio", f"{iv_hv_ratio_20d:.2f}x")
                        
                        if iv_hv_ratio_20d > 1.2:
                            st.warning("🔺 IV elevated vs recent history")
                        elif iv_hv_ratio_20d < 0.8:
                            st.info("🔻 IV depressed vs recent history")
                        else:
                            st.success("⚖️ IV aligned with historical levels")
                    
                except (ValueError, RuntimeError) as e:
                    st.error("❌ Could not calculate implied volatility")
                    st.caption("The market price may be outside reasonable bounds or inconsistent with current parameters")
            else:
                st.warning("Enter a market price > 0 to calculate IV")
        
        # Additional analysis
        if market_price > 0:
            st.markdown("---")
            st.markdown("### 📈 Greeks at Calculated IV")
            
            try:
                delta, gamma, vega, theta, rho = greeks(S, K, T, r, implied_vol, option_type)
                
                greeks_data = {
                    "Greek": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
                    "Value": [f"{delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{theta:.4f}", f"{rho:.4f}"],
                    "Interpretation": [
                        f"${delta:.2f} change per $1 stock move",
                        f"Delta changes by {gamma:.4f} per $1 move",
                        f"${vega:.2f} change per 1% vol increase",
                        f"${theta:.2f} daily time decay",
                        f"${rho:.2f} change per 1% rate increase"
                    ]
                }
                
                greeks_df = pd.DataFrame(greeks_data)
                st.dataframe(greeks_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error calculating Greeks: {str(e)}")

if __name__ == "__main__":
    iv_analysis_page()
