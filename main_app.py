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
from mistralai import Mistral
import os

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
    
    # --- Japanese Market News Section (Mistral AI Summarized) ---
    
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    
    # --- Configuration: Japanese Financial RSS Feeds (Yahoo Japan Media) ---
    JAPANESE_FINANCIAL_RSS_FEEDS = {
        "Forbes Japan": "https://news.yahoo.co.jp/rss/media/forbes/all.xml",
        "Diamond Online": "https://news.yahoo.co.jp/rss/media/dzai/all.xml",
        "Teikoku Databank": "https://news.yahoo.co.jp/rss/media/teikokudb/all.xml",
        "Yahoo Business": "https://news.yahoo.co.jp/rss/media/business/all.xml",
        "Finasee": "https://news.yahoo.co.jp/rss/media/finasee/all.xml"
    }
    
    RSS_DESCRIPTIONS = {
        "Forbes Japan": "Business and finance news with insights on companies and markets.",
        "Diamond Online": "Economic and financial news from a leading Japanese business magazine.",
        "Teikoku Databank": "Corporate credit reports, market trends, and industry insights.",
        "Yahoo Business": "General business news and stock market updates in Japan.",
        "Finasee": "Finance-focused news with emphasis on stocks, investments, and market trends."
    }
    
    # --- Fetch RSS with caching ---
    @st.cache_data(show_spinner=False)
    def fetch_news_rss(rss_urls, top_n=30):
        import feedparser
        all_entries = []
        for url in rss_urls:
            feed = feedparser.parse(url)
            all_entries.extend(feed.entries)
        # Sort by published date if available
        all_entries.sort(key=lambda x: getattr(x, "published_parsed", None), reverse=True)
        return all_entries[:top_n]
    
    # --- Summarization using Mistral AI ---
    def summarize_news_mistral(news_entries):
        # Prepare combined prompt
        combined_text = ""
        for entry in news_entries:
            combined_text += f"Title: {entry.title}\nSummary: {getattr(entry, 'summary', '')}\nLink: {entry.link}\n\n"
    
        prompt = (
            "You are an AI financial journalist. From the following Japanese business and market news, "
            "choose the most pertinent items, summarize each in 2-3 sentences, "
            "and structure the output clearly with the title, short summary, and link to the original article. "
            "Prioritize relevance to markets, finance, and corporate news:\n\n"
            f"{combined_text}"
        )
    
        try:
            client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
            chat_response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary unavailable: {str(e)}"
    
    # --- Streamlit Interface ---
    st.subheader("📰 Japanese Market News (AI-Summarized)")
    st.markdown(
        "Latest financial news from Japanese Yahoo Japan media sources, merged and summarized by Mistral AI Experiment* "
        "(*free tier, limited requests)."
    )
    
    # Display sources and description
    st.markdown("**Sources included:**")
    for src, desc in RSS_DESCRIPTIONS.items():
        st.markdown(f"- **{src}**: {desc}")
    
    # Fetch and merge feeds
    news_items = fetch_news_rss(list(JAPANESE_FINANCIAL_RSS_FEEDS.values()), top_n=30)
    
    # Generate AI summary
    if news_items:
        ai_summary = summarize_news_mistral(news_items)
        st.markdown(ai_summary)
    else:
        st.info("No news available at the moment.")

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
