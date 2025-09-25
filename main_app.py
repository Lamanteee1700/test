import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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
            "USD/EUR": "EURUSD=X"
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
            "USD/EUR": {"value": 1.085, "change": 0.3}
        }

# --- Market Overview ---
st.subheader("📊 Live Market Overview")

market_data = get_market_data()

col1, col2, col3, col4 = st.columns(4)

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
    eur_data = market_data["USD/EUR"]
    col4.metric("USD/EUR", f"{eur_data['value']:.3f}", f"{eur_data['change']:+.2f}%")

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
                and market price comparison capabilities.
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Black-Scholes pricing</li>
                    <li>Yahoo Finance integration</li>
                    <li>Interactive Greeks</li>
                    <li>Implied volatility</li>
                </ul>
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
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Complex portfolios</li>
                    <li>Dynamic hedging</li>
                    <li>Institutional presets</li>
                    <li>P&L visualization</li>
                </ul>
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
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Historical analysis</li>
                    <li>Volatility cone</li>
                    <li>Smile/Skew modeling</li>
                    <li>Monte Carlo simulation</li>
                </ul>
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
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Yield curve modeling</li>
                    <li>Fair swap rate calculation</li>
                    <li>DV01 analysis</li>
                    <li>Scenario stress testing</li>
                </ul>
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
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Monte Carlo pricing</li>
                    <li>Component breakdown</li>
                    <li>Risk analysis</li>
                    <li>Stress scenarios</li>
                </ul>
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
                <br><br>
                <strong>Key Features:</strong>
                <ul>
                    <li>Structural models</li>
                    <li>Portfolio VaR</li>
                    <li>Stress testing</li>
                    <li>Regulatory capital</li>
                </ul>
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
                <br><br>
                <strong>News Sources:</strong>
                <ul>
                    <li>Forbes Japan</li>
                    <li>Diamond Online</li>
                    <li>Yahoo Finance Japan</li>
                    <li>Mistral AI summaries</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with market_col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📈</span>
            <h4 class="feature-title">Technical Analysis Suite</h4>
            <div class="feature-description">
                Comprehensive technical analysis tools with advanced 
                indicators, strategy backtesting, and quantitative 
                market screening capabilities.
                <br><br>
                <strong>In Development:</strong>
                <ul>
                    <li>Advanced indicators</li>
                    <li>Backtesting engine</li>
                    <li>Stock screening</li>
                    <li>Automated alerts</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Platform statistics ---
st.subheader("📊 Platform Statistics")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.markdown("""
    <div class="highlight-metric">
        <h3>12+</h3>
        <p>Analysis Tools</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown("""
    <div class="highlight-metric">
        <h3>5+</h3>
        <p>Asset Classes</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown("""
    <div class="highlight-metric">
        <h3>Real-time</h3>
        <p>Market Data</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col4:
    st.markdown("""
    <div class="highlight-metric">
        <h3>24/7</h3>
        <p>Availability</p>
    </div>
    """, unsafe_allow_html=True)

# --- Navigation guide ---
with st.expander("🧭 Navigation Guide", expanded=False):
    st.markdown("""
    ### How to use this platform:
    
    **1. Beginners:** Start with the **Options Pricer** to understand derivatives pricing fundamentals.
    
    **2. Intermediate:** Explore **Volatility Strategies** and **Greeks Hedging** for sophisticated approaches.
    
    **3. Advanced:** Use **Structured Products** and **Credit Risk** tools for institutional-grade analysis.
    
    **4. Market Intelligence:** Stay informed with **Japanese Financial News** powered by AI.
    
    ### Key Features:
    - **Real-time data** via Yahoo Finance API
    - **Interactive visualizations** with Plotly
    - **Academic models** (Black-Scholes, Merton, etc.)
    - **Institutional use cases** (banks, funds, insurance)
    - **Professional interface** designed for practitioners
    
    ### Recommended Learning Path:
    1. **Options Pricer** → Understanding derivatives fundamentals
    2. **Volatility Strategies** → Advanced option strategies  
    3. **Greeks Hedging** → Risk management techniques
    4. **Swaps Pricer** → Fixed income analytics
    5. **Credit Risk** → Portfolio risk management
    6. **Structured Products** → Complex instrument design
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

# --- Use cases by institution type ---
with st.expander("🏛️ Institutional Use Cases", expanded=False):
    
    usecase_col1, usecase_col2 = st.columns(2)
    
    with usecase_col1:
        st.markdown("""
        **Investment Banks:**
        - Options market making and pricing
        - Structured products design
        - Risk management and hedging
        - Regulatory capital calculations
        
        **Asset Managers:**
        - Portfolio optimization
        - Risk budgeting and attribution
        - Factor exposure analysis
        - Performance measurement
        """)
    
    with usecase_col2:
        st.markdown("""
        **Insurance Companies:**
        - Asset-liability matching
        - Solvency II compliance
        - Credit risk assessment
        - Structured product evaluation
        
        **Pension Funds:**
        - Long-term portfolio construction
        - Interest rate risk hedging
        - Alternative investment analysis
        - ESG integration
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

# --- Welcome animation (optional) ---
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
    st.balloons()
