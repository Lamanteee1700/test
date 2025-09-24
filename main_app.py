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
import feedparser
import requests
import random

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
    
    # Create feature cards
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
    
    # Fetch some market data for context
    try:
        # Get major indices
        tickers = ["^GSPC", "^VIX", "^TNX"]  # S&P 500, VIX, 10Y Treasury
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
            market_df = pd.DataFrame(market_data)
            
            # Display market data
            mcol1, mcol2, mcol3 = st.columns(3)
            
            for i, (col, row) in enumerate(zip([mcol1, mcol2, mcol3], market_data)):
                with col:
                    if i == 0:  # S&P 500
                        col.metric(row["Index"], f"{row['Value']:.0f}", f"{row['Change %']:+.2f}%")
                    elif i == 1:  # VIX
                        col.metric(row["Index"], f"{row['Value']:.1f}", f"{row['Change %']:+.2f}%")
                    else:  # 10Y Treasury
                        col.metric(row["Index"], f"{row['Value']:.2f}%", f"{row['Change %']:+.2f}%")
        
    except Exception as e:
        st.info("Market data temporarily unavailable")
    
    # Navigation Instructions
    st.subheader("🧭 Navigation Guide")
    
    st.markdown("""
    **Use the sidebar navigation to explore different tools:**
    
    - **📊 1_Options_pricer** - Advanced options pricing with company search and comprehensive analytics
    - **📖 2_Theory** - Black-Scholes theory, assumptions, and mathematical foundations  
    - **📐 3_Options_combinations** - Build and analyze option strategies (straddles, spreads, etc.)
    - **🧮 4_Greeks_hedging_strategies** - Delta and gamma hedging demonstrations
    - **📈 5_volatility_strategies** - VIX analysis and volatility surface exploration
    - **💰 8_swaps_pricer** - Interest rate swap pricing and risk analysis
    - **🏗️ 9_Structured_products_builder** - Build complex structured derivative products
    
    Each page provides interactive tools with educational explanations to help you understand derivative pricing and risk management.
    """)
    
    # Educational Resources
    with st.expander("📚 Educational Resources & Key Concepts"):
        st.markdown("""
        ### Core Financial Derivatives Concepts
        
        **Options Fundamentals:**
        - **Call Options**: Right to buy at strike price - profit when price rises
        - **Put Options**: Right to sell at strike price - profit when price falls  
        - **Greeks**: Risk sensitivities (Delta, Gamma, Vega, Theta, Rho)
        - **Implied Volatility**: Market's expectation of future price movements
        
        **Interest Rate Swaps:**
        - **Fixed vs Floating**: Exchange fixed-rate payments for floating-rate payments
        - **Fair Swap Rate**: Rate that makes both legs equal in value at inception
        - **Duration Risk**: Sensitivity to parallel shifts in yield curves
        - **Curve Risk**: Sensitivity to changes in yield curve shape
        
        **Structured Products:**
        - **Capital Protection**: Guaranteed return of principal (or percentage)
        - **Enhanced Yield**: Higher coupons in exchange for conditional risks
        - **Path Dependency**: Payoffs depend on price evolution, not just final value
        - **Barrier Features**: Knock-in/knock-out conditions that activate/deactivate payoffs
        
        ### Risk Management Applications
        - **Hedging**: Reduce unwanted risk exposures
        - **Speculation**: Express market views with defined risk/reward profiles
        - **Asset-Liability Matching**: Align investment characteristics with obligations
        - **Yield Enhancement**: Generate additional income through option strategies
        """)
    
    # Technical Notes
    with st.expander("⚙️ Technical Implementation Notes"):
        st.markdown("""
        **Models & Methods:**
        - **Black-Scholes-Merton**: European option pricing with analytical Greeks
        - **Monte Carlo Simulation**: Path-dependent and barrier option pricing  
        - **Present Value Methods**: Bond and swap valuation using yield curves
        - **Numerical Methods**: Finite difference methods for complex derivatives
        
        **Data Sources:**
        - **Yahoo Finance API**: Real-time equity prices and market data
        - **Synthetic Curves**: Modeled yield curves for educational purposes
        - **Historical Volatility**: Calculated from price return series
        
        **Limitations & Assumptions:**
        - Educational tool - not for actual trading decisions
        - Simplified models may not capture all market complexities
        - Real-world factors: bid-ask spreads, liquidity, early exercise features
        - Regulatory and tax considerations not included
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
    <strong>Financial Derivatives Dashboard</strong><br>
    Educational platform for options, swaps, and structured products analysis<br>
    Built with Streamlit, NumPy, SciPy, Plotly & Yahoo Finance
    </div>
    """, unsafe_allow_html=True)
    
    # --- Japanese Market News Section (Mistral AI Summarized) ---
    
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    MISTRAL_URL = "https://api.mistral.ai/v1/generate"
    
    
    # Liste des sources RSS japonaises pour les marchés financiers
    JAPANESE_FINANCIAL_RSS_FEEDS = [
        "https://news.yahoo.co.jp/rss/topics/business.xml",  # Yahoo Japan Finance
        "https://asia.nikkei.com/rss",                      # Nikkei Asian Review
        "https://www.japantimes.co.jp/feed/business/",      # Japan Times Business
        "https://jp.reuters.com/tools/rss",                # Reuters Japan
    ]
    
    @st.cache_data(show_spinner=False)
    def fetch_news_rss(rss_url=None, top_n=10):
        if rss_url is None:
            rss_url = random.choice(JAPANESE_FINANCIAL_RSS_FEEDS)
        feed = feedparser.parse(rss_url)
        return feed.entries[:top_n]
    
    @st.cache_data(show_spinner=False)
    def summarize_news_mistral(title, summary=""):
        prompt = f"Summarize this Japanese financial news in 2-3 sentences, keeping the key points:\nTitle: {title}\nSummary: {summary}"
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
        payload = {
            "model": "mistral-large",
            "prompt": prompt,
            "max_tokens": 200
        }
        try:
            response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if "text" in result:
                    return result["text"].strip()
                else:
                    return "Summary unavailable"
            else:
                return f"API Error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    # Fetch latest news
    news_items = []
    for feed_url in JAPANESE_FINANCIAL_RSS_FEEDS:
        try:
            news_items.extend(fetch_news_rss(feed_url, top_n=3))
        except Exception as e:
            st.warning(f"Could not fetch news from {feed_url}: {str(e)}")
    
    # Display in Streamlit
    st.subheader("📰 Japanese Market News (AI-Summarized)")
    
    if news_items:
        for entry in news_items[:15]:
            title = entry.title
            summary_text = getattr(entry, "summary", "")
            ai_summary = summarize_news_mistral(title, summary_text)
    
            st.markdown(f"- [{title}]({entry.link})")
            st.markdown(f"  *{ai_summary}*")
    else:
        st.info("No news available at the moment.")
    
if __name__ == "__main__":
    main()
