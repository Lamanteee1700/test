import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- Sidebar with LinkedIn ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 👤 Developer")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <a href="https://www.linkedin.com/in/c%C3%B4me-kasai-campagnolo-765b39244/" target="_blank" 
           style="display: inline-block; background: #0077B5; color: white; padding: 0.5rem 1rem; 
                  border-radius: 5px; text-decoration: none; font-weight: 500;">
            🔗 LinkedIn Profile
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Financial Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for professional styling ---
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.main-title {
    color: white;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-subtitle {
    color: #e6f3ff;
    font-size: 1.2rem;
    margin-bottom: 0;
    opacity: 0.95;
}

.feature-card {
    background: white;
    border: 1px solid #e0e6ed;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 280px;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}

.feature-title {
    color: #2c3e50;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.feature-description {
    color: #5a6c7d;
    line-height: 1.6;
    font-size: 0.95rem;
}

.stats-card {
    background: linear-gradient(45deg, #f8f9fa, #ffffff);
    border-left: 4px solid #667eea;
    padding: 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.market-overview {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
}

.highlight-metric {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}

.navigation-hint {
    background: linear-gradient(135deg, #ffeaa7, #fab1a0);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    border-left: 4px solid #e17055;
}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🏦 Advanced Financial Analytics Platform</h1>
    <p class="main-subtitle">
        Professional suite of tools for derivatives analysis, portfolio construction, 
        and institutional risk management
    </p>
    <div style="background: rgba(255,193,7,0.2); padding: 0.8rem; border-radius: 8px; margin-top: 1rem;">
        <strong>⚠️ Work in Progress:</strong> This platform is currently under active development. 
        Features are being continuously improved and expanded.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Real-time market metrics ---
def get_market_data():
    """Fetch real-time market data"""
    try:
        tickers = {
            "S&P 500": "^GSPC",
            "VIX": "^VIX", 
            "10Y Treasury": "^TNX",
            "USD/JPY": "USDJPY=X",
            "Nikkei 225": "^N225",
            "JPY 10Y Bond": "^TNX"  # Placeholder for Japan 10Y
        }
        
        market_data = {}
        for name, ticker in tickers.items():
            try:
                data = yf.Ticker(ticker).history(period="2d")
                if not data.empty:
                    current = data["Close"].iloc[-1]
                    prev = data["Close"].iloc[-2] if len(data) > 1 else current
                    change_pct = (current - prev) / prev * 100
                    market_data[name] = {"value": current, "change": change_pct}
            except:
                market_data[name] = {"value": 0, "change": 0}
        
        return market_data
    except:
        return {
            "S&P 500": {"value": 4500, "change": 0.5},
            "VIX": {"value": 18.5, "change": -2.1},
            "10Y Treasury": {"value": 4.25, "change": 0.1},
            "USD/JPY": {"value": 148.5, "change": 0.3},
            "Nikkei 225": {"value": 38500, "change": 1.2},
            "JPY 10Y Bond": {"value": 1.1, "change": 0.05}
        }

# --- Market Overview ---
st.subheader("📊 Live Market Overview")

market_data = get_market_data()

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    sp_data = market_data["S&P 500"]
    col1.metric("S&P 500", f"{sp_data['value']:.0f}", f"{sp_data['change']:+.2f}%")

with col2:
    vix_data = market_data["VIX"]
    col2.metric("VIX", f"{vix_data['value']:.1f}", f"{vix_data['change']:+.2f}%")

with col3:
    treasury_data = market_data["10Y Treasury"]
    col3.metric("10Y US Treasury", f"{treasury_data['value']:.2f}%", f"{treasury_data['change']:+.2f}%")

with col4:
    jpy_data = market_data["USD/JPY"]
    col4.metric("USD/JPY", f"{jpy_data['value']:.1f}", f"{jpy_data['change']:+.2f}%")

with col5:
    nikkei_data = market_data["Nikkei 225"]
    col5.metric("Nikkei 225", f"{nikkei_data['value']:.0f}", f"{nikkei_data['change']:+.2f}%")

with col6:
    jpy_bond_data = market_data["JPY 10Y Bond"]
    col6.metric("JPY 10Y Bond", f"{jpy_bond_data['value']:.2f}%", f"{jpy_bond_data['change']:+.2f}%")

# --- Navigation hint ---
st.markdown("""
<div class="navigation-hint">
    <strong>🧭 Quick Navigation:</strong> Use the sidebar to explore our comprehensive financial tools, 
    from basic option pricing to advanced institutional risk management frameworks.
</div>
""", unsafe_allow_html=True)

# --- Main navigation by modules ---
st.subheader("🎯 Financial Analysis Modules")

# Organized in tabs for better structure
tab_derivatives, tab_portfolio, tab_markets = st.tabs([
    "📊 Derivatives & Options", 
    "🏛️ Portfolio Construction", 
    "📈 Markets & Analysis"
])

with tab_derivatives:
    st.markdown("### Derivatives Analysis Tools")
    
    der_col1, der_col2, der_col3 = st.columns(3)
    
    with der_col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <h4 class="feature-title">Advanced Options Pricer</h4>
            <div class="feature-description">
                Black-Scholes model with live market data integration, 
                comprehensive Greeks analysis, sensitivity testing, 
                and market price comparison.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with der_col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧮</span>
            <h4 class="feature-title">Multi-Asset Greeks Strategy</h4>
            <div class="feature-description">
                Sophisticated multi-asset portfolio construction with 
                advanced Greeks management, delta hedging, gamma hedging, 
                and correlation analysis tools.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with der_col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🌊</span>
            <h4 class="feature-title">Volatility Strategies</h4>
            <div class="feature-description">
                Advanced volatility analysis with VIX integration, 
                volatility surfaces, trading simulation, and 
                comprehensive vol strategy backtesting.
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_portfolio:
    st.markdown("### Portfolio Construction Tools")
    
    port_col1, port_col2, port_col3 = st.columns(3)
    
    with port_col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💰</span>
            <h4 class="feature-title">Interest Rate Swaps Pricer</h4>
            <div class="feature-description">
                Complete swap valuation with dynamic yield curves, 
                comprehensive sensitivity analysis, and sophisticated 
                interest rate risk management tools.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with port_col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🏗️</span>
            <h4 class="feature-title">Structured Products Builder</h4>
            <div class="feature-description">
                Design and analyze complex structured products: 
                reverse convertibles, autocallables, capital protection, 
                and barrier options with full risk decomposition.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with port_col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🏦</span>
            <h4 class="feature-title">Credit Risk Management</h4>
            <div class="feature-description">
                Advanced credit risk frameworks with Merton models, 
                transition matrices, portfolio VaR, and comprehensive 
                Basel III regulatory capital calculations.
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_markets:
    st.markdown("### Market Analysis & Intelligence")
    
    market_col1, market_col2 = st.columns(2)
    
    with market_col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📰</span>
            <h4 class="feature-title">Japanese Market News</h4>
            <div class="feature-description">
                Automated RSS feeds from major Japanese financial media 
                with AI-powered summaries and sentiment analysis for 
                comprehensive market intelligence.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Upcoming Features ---
st.subheader("🚀 Coming Soon")

upcoming_col1, upcoming_col2, upcoming_col3 = st.columns(3)

with upcoming_col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <h4 class="feature-title">Fixed Income Strategies</h4>
        <div class="feature-description">
            Comprehensive fixed income portfolio construction 
            with yield curve modeling, duration management, 
            and credit risk analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

with upcoming_col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🏗️</span>
        <h4 class="feature-title">Commodities Trading</h4>
        <div class="feature-description">
            Advanced commodities analysis with futures 
            pricing models, seasonality patterns, and 
            supply-demand dynamics.
        </div>
    </div>
    """, unsafe_allow_html=True)

with upcoming_col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🧮</span>
        <h4 class="feature-title">Advanced Models</h4>
        <div class="feature-description">
            Next-generation quantitative models including 
            Monte Carlo simulations, Jump Diffusion, 
            and stochastic volatility frameworks.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Navigation guide ---
with st.expander("🧭 Navigation Guide", expanded=False):
    st.markdown("""
    ### How to use this platform:
    
    **1. Beginners:** Start with the **Options Pricer** to understand derivatives pricing fundamentals.
    
    **2. Intermediate:** Explore **Volatility Strategies** and **Greeks Hedging** for sophisticated approaches.
    
    **3. Advanced:** Use **Structured Products** and **Credit Risk** tools for institutional-grade analysis.
    
    ### Key Features:
    - **Real-time data** via Yahoo Finance API
    - **Interactive visualizations** with Plotly
    - **Academic models** (Black-Scholes, Merton, etc.)
    - **Professional interface** designed for practitioners
    """)

# --- Technologies & methodologies ---
with st.expander("⚙️ Technologies & Methodologies", expanded=False):
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Technology Stack:**
        - **Streamlit** - Interactive web interface
        - **Python** - Financial computations
        - **Plotly** - Advanced visualizations
        - **Yahoo Finance API** - Real-time market data
        - **SciPy** - Numerical optimization
        - **NumPy/Pandas** - Matrix computations
        - **Mistral AI** - News summarization
        """)
    
    with tech_col2:
        st.markdown("""
        **Financial Models:**
        - **Black-Scholes-Merton** - Options pricing
        - **Merton Models** - Credit risk
        - **Monte Carlo** - Path-dependent pricing
        - **VaR/CVaR** - Risk measurement
        - **Greeks** - Sensitivity analysis
        - **Yield Curves** - Interest rate modeling
        - **Factor Models** - Portfolio construction
        """)

# --- Professional footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 2rem;">
    <h4 style="color: #2c3e50; margin-bottom: 1rem;">🏦 Advanced Financial Analytics Platform</h4>
    <p style="color: #6c757d; margin-bottom: 0;">
        Professional educational platform for complex financial instrument analysis<br>
        <strong>Streamlit • Python • Quantitative Finance • Real-Time Data</strong><br>
        <em>Last updated: {}</em>
    </p>
    <br>
    <div style="font-size: 0.9rem; color: #868e96;">
        <strong>Disclaimer:</strong> This platform is designed for educational and research purposes. 
        All models and calculations should be validated before use in production trading or risk management systems.
    </div>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
