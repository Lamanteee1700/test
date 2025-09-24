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
import feedparser  # Added for RSS news parsing

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Derivatives Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Dashboard Landing Page ---
def main():
    st.title("📊 Financial Derivatives Dashboard")
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 15px; border-left: 5px solid #1f77b4; margin: 20px 0;">
    <h3 style="margin: 0; color: #1f77b4;">Welcome to the Financial Derivatives Analytics Platform</h3>
    <p style="margin: 10px 0 0 0;">A comprehensive educational tool for options pricing, swaps analysis, and structured products.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Overview
    st.subheader("🚀 Platform Features")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745; height: 200px;">
        <h4 style="color: #28a745; margin-top: 0;">🎯 Options Pricer</h4>
        <ul style="color: #333;">
        <li>Black-Scholes pricing & Greeks</li>
        <li>Advanced analytics & visualizations</li>
        <li>Real-time market data integration</li>
        <li>Implied volatility calculations</li>
        <li>Company search & information</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #007bff; height: 200px;">
        <h4 style="color: #007bff; margin-top: 0;">💰 Swaps Pricer</h4>
        <ul style="color: #333;">
        <li>Interest rate swap valuation</li>
        <li>Yield curve analysis</li>
        <li>Cash flow visualization</li>
        <li>Sensitivity & risk metrics</li>
        <li>Fair swap rate calculation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #dc3545; height: 200px;">
        <h4 style="color: #dc3545; margin-top: 0;">🏗️ Structured Products</h4>
        <ul style="color: #333;">
        <li>Reverse convertibles</li>
        <li>Autocallable notes</li>
        <li>Capital protected notes</li>
        <li>Barrier options analysis</li>
        <li>Monte Carlo simulations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Market Overview
    st.subheader("📈 Market Snapshot")
    
    try:
        tickers = ["^GSPC", "^VIX", "^TNX"]
        ticker_names = ["S&P 500", "VIX", "10Y Treasury"]
        
        market_data = []
        for ticker, name in zip(tickers, ticker_names):
            try:
                data = yf.Ticker(ticker).history(period="2d")
                if not data.empty:
                    current = data["Close"].iloc[-1]
                    prev = data["Close"].iloc[-2] if len(data) > 1 else current
                    change_pct = (current - prev) / prev * 100
                    market_data.append({
                        "Index": name,
                        "Value": current,
                        "Change %": change_pct
                    })
            except:
                continue
        
        if market_data:
            mcol1, mcol2, mcol3 = st.columns(3)
            for i, (col, row) in enumerate(zip([mcol1, mcol2, mcol3], market_data)):
                with col:
                    if i == 0:
                        col.metric(row["Index"], f"{row['Value']:.0f}", f"{row['Change %']:+.2f}%")
                    elif i == 1:
                        col.metric(row["Index"], f"{row['Value']:.1f}", f"{row['Change %']:+.2f}%")
                    else:
                        col.metric(row["Index"], f"{row['Value']:.2f}%", f"{row['Change %']:+.2f}%")
        
    except Exception as e:
        st.info("Market data temporarily unavailable")
    
    # Navigation Instructions
    st.subheader("🧭 Navigation Guide")
    
    st.markdown("""
    **Use the sidebar navigation to explore different tools:**
    
    - **📊 1_Options_pricer** - Advanced options pricing with company search and analytics
    - **📖 2_Theory** - Black-Scholes theory and foundations  
    - **📐 3_Options_combinations** - Build and analyze option strategies
    - **🧮 4_Greeks_hedging_strategies** - Delta and gamma hedging demos
    - **📈 5_volatility_strategies** - VIX analysis and volatility surfaces
    - **💰 8_swaps_pricer** - Interest rate swap pricing & risk analysis
    - **🏗️ 9_Structured_products_builder** - Build structured derivative products
    
    Each page provides interactive tools with educational explanations to understand derivative pricing and risk.
    """)
    
    # Educational Resources
    with st.expander("📚 Educational Resources & Key Concepts"):
        st.markdown("""
        ### Core Financial Derivatives Concepts
        **Options Fundamentals:**
        - Call / Put, Greeks, Implied Volatility  
        **Interest Rate Swaps:**
        - Fixed vs Floating, Fair Swap Rate, Duration & Curve Risk  
        **Structured Products:**
        - Capital Protection, Enhanced Yield, Path Dependency, Barrier Features  
        ### Risk Management Applications
        - Hedging, Speculation, Asset-Liability Matching, Yield Enhancement
        """)
    
    # Technical Notes
    with st.expander("⚙️ Technical Implementation Notes"):
        st.markdown("""
        **Models & Methods:** Black-Scholes-Merton, Monte Carlo, PV Methods, Numerical Methods  
        **Data Sources:** Yahoo Finance, Synthetic Curves, Historical Volatility  
        **Limitations:** Educational tool, simplified models, no trading advice
        """)
    
    # --- Japanese Market News Section ---
    st.subheader("📰 Japanese Market News")
    st.markdown("Stay up-to-date with top financial and market headlines in Japan:")
    
    rss_url = "https://news.yahoo.co.jp/rss/topics/business.xml"
    try:
        feed = feedparser.parse(rss_url)
        news_items = feed.entries[:5]  # Top 5 articles
        for entry in news_items:
            st.markdown(f"- [{entry.title}]({entry.link}) ({entry.published})")
    except Exception as e:
        st.info("Japanese market news temporarily unavailable.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
    <strong>Financial Derivatives Dashboard</strong><br>
    Educational platform for options, swaps, and structured products analysis<br>
    Built with Streamlit, NumPy, SciPy, Plotly & Yahoo Finance
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
