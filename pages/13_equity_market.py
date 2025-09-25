# pages/13_Equity_fundamentals.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def dcf_model(free_cash_flows, terminal_growth, discount_rate):
    """
    Calculate intrinsic value using DCF model
    """
    pv_cash_flows = []
    for i, fcf in enumerate(free_cash_flows):
        pv = fcf / (1 + discount_rate) ** (i + 1)
        pv_cash_flows.append(pv)
    
    # Terminal value
    terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** len(free_cash_flows)
    
    enterprise_value = sum(pv_cash_flows) + pv_terminal
    
    return {
        'pv_cash_flows': pv_cash_flows,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value
    }

def calculate_equity_risk_premium(risk_free_rate, market_return):
    """Calculate equity risk premium"""
    return market_return - risk_free_rate

def simulate_compound_returns(initial_investment, annual_return, years, volatility=0.15):
    """Simulate compound returns with volatility"""
    np.random.seed(42)
    
    values = [initial_investment]
    for year in range(years):
        annual_vol_return = np.random.normal(annual_return, volatility)
        new_value = values[-1] * (1 + annual_vol_return)
        values.append(new_value)
    
    return values

def calculate_factor_returns():
    """Generate sample factor performance data"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    
    # Simulate factor returns
    np.random.seed(42)
    
    factors = {
        'Market': np.random.normal(0.008, 0.04, len(dates)),  # 0.8% monthly avg
        'Value': np.random.normal(0.002, 0.03, len(dates)),   # Value factor
        'Growth': np.random.normal(0.006, 0.05, len(dates)),  # Growth factor  
        'Quality': np.random.normal(0.004, 0.025, len(dates)), # Quality factor
        'Low Vol': np.random.normal(0.003, 0.02, len(dates)),  # Low volatility
        'Momentum': np.random.normal(0.005, 0.06, len(dates))  # Momentum
    }
    
    # Calculate cumulative returns
    factor_cumulative = {}
    for factor, returns in factors.items():
        cumulative = np.cumprod(1 + returns) - 1
        factor_cumulative[factor] = cumulative
    
    return dates, factor_cumulative

def create_sector_allocation_data():
    """Create sample sector allocation data"""
    sectors = [
        'Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
        'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
        'Utilities', 'Real Estate', 'Materials'
    ]
    
    # Sample market cap weights (approximate S&P 500)
    weights = [28.5, 13.1, 12.9, 10.8, 8.7, 8.2, 6.1, 4.2, 2.8, 2.5, 2.2]
    
    return sectors, weights

def calculate_pe_ratio_history():
    """Generate sample P/E ratio historical data"""
    dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='Q')
    
    # Create cyclical P/E ratios with long-term trends
    base_pe = 16
    pe_ratios = []
    
    for i, date in enumerate(dates):
        # Long-term trend
        trend = 0.02 * i / len(dates)  # Slight upward trend
        
        # Business cycle
        cycle = 4 * np.sin(2 * np.pi * i / 40)  # ~10 year cycle
        
        # Random noise
        noise = np.random.normal(0, 1)
        
        # Crisis periods (2008, 2020)
        if (date.year == 2008 and date.month >= 9) or (date.year == 2009):
            crisis = -8
        elif date.year == 2020 and date.month >= 3 and date.month <= 6:
            crisis = -5
        else:
            crisis = 0
        
        pe = base_pe + trend + cycle + noise + crisis
        pe_ratios.append(max(8, pe))  # Floor at 8
    
    return dates, pe_ratios

def show_equity_fundamentals_page():
    st.set_page_config(page_title="Equity Fundamentals", layout="wide")
    
    st.title("📈 Equity Fundamentals: Ownership, Valuation & Market Dynamics")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Understanding Stocks: From Ownership Rights to Market Valuation</h3>
        <p style="color: #e8f4fd; margin: 0.5rem 0 0 0;">
            Master equity fundamentals from basic ownership concepts to advanced valuation models and factor investing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === SECTION 1: WHAT IS AN EQUITY? ===
    st.subheader("🏢 What an Equity Actually Represents")
    
    with st.expander("📊 Equity as Residual Ownership", expanded=True):
        st.markdown("""
        **Equity** represents ownership in a corporation - a residual claim on the company's assets and earnings 
        after all debt obligations have been satisfied. Unlike bondholders who have contractual claims, 
        equity holders own a proportional share of the company's future success or failure.
        
        ### Key Ownership Rights:
        
        **1. Voting Rights**
        - Elect board of directors
        - Approve major corporate decisions (mergers, acquisitions)
        - Vote on executive compensation and auditor selection
        - Influence strategic direction through shareholder proposals
        
        **2. Dividend Rights**
        - Claim on distributed profits (when declared by board)
        - No contractual obligation - dividends can be cut or suspended
        - Common shareholders receive dividends after preferred shareholders
        
        **3. Liquidation Rights**
        - Residual claim on assets after debt and preferred equity
        - Last in line during bankruptcy proceedings
        - Can lose entire investment if liabilities exceed assets
        
        ### Capital Structure Hierarchy:
        
        **Payment Priority (Crisis/Bankruptcy):**
        1. **Secured Debt** (highest priority)
        2. **Senior Unsecured Debt**
        3. **Subordinated Debt**
        4. **Preferred Equity**
        5. **Common Equity** (lowest priority - residual claim)
        
        This hierarchy explains why equity is inherently riskier than debt but offers unlimited upside potential.
        """)
    
    # Capital Structure Visualization
    with st.container():
        st.markdown("#### Capital Structure: Risk vs Return Profile")
        
        # Create capital structure chart
        securities = ['Senior Debt', 'Subordinated Debt', 'Preferred Stock', 'Common Stock']
        risk_levels = [2, 4, 6, 9]
        return_potential = [4, 6, 7, 12]
        priority_order = [1, 2, 3, 4]
        
        fig_capital = go.Figure()
        
        colors = ['green', 'yellow', 'orange', 'red']
        for i, (security, risk, ret, priority) in enumerate(zip(securities, risk_levels, return_potential, priority_order)):
            fig_capital.add_trace(go.Scatter(
                x=[risk], y=[ret],
                mode='markers+text',
                name=security,
                text=[f"{security}<br>Priority: {priority}"],
                textposition="middle center",
                marker=dict(size=80, color=colors[i], opacity=0.7),
                showlegend=False
            ))
        
        fig_capital.update_layout(
            title="Risk-Return Profile Across Capital Structure",
            xaxis_title="Risk Level",
            yaxis_title="Expected Return (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_capital, use_container_width=True)
    
    with st.expander("🎯 Equity as an Embedded Call Option"):
        st.markdown("""
        ### The Option Perspective on Equity
        
        From a theoretical standpoint, **equity can be viewed as a call option on the firm's assets** 
        with the debt value as the strike price. This perspective, developed by Black, Scholes, and Merton, 
        provides crucial insights into equity behavior.
        
        **Key Insights from Option Theory:**
        
        **1. Limited Downside, Unlimited Upside**
        - Equity holders cannot lose more than their investment (like option buyers)
        - Unlimited participation in company growth
        - Explains why high-leverage companies have volatile equity prices
        
        **2. Volatility Creates Value** 
        - Higher business volatility increases option value
        - Explains why growth/tech stocks command premiums
        - Risk can actually benefit equity holders at expense of debt holders
        
        **3. Time Value Effects**
        - Companies near bankruptcy may have equity value due to "time value"
        - Potential for business turnarounds keeps equity prices above zero
        - Similar to out-of-the-money options with time to expiration
        
        **4. Interest Rate Sensitivity**
        - Lower rates increase present value of future cash flows
        - Reduces the "cost" of carrying the equity position
        - Explains why tech/growth stocks are rate-sensitive
        
        This framework helps explain seemingly irrational equity prices during distressed periods 
        and the high valuations of loss-making growth companies.
        """)
    
    # === SECTION 2: LONG-TERM CASE FOR EQUITIES ===
    st.subheader("📊 The Long-Term Case for Equities")
    
    st.markdown("""
    **Historical evidence** strongly supports equity ownership as a wealth-building strategy over long time horizons, 
    despite short-term volatility and periodic drawdowns.
    """)
    
    # Historical Returns Simulation
    with st.container():
        st.markdown("#### Historical Asset Class Performance (Simulated)")
        
        # Simulate long-term returns
        years = 30
        initial_investment = 10000
        
        # Historical average annual returns (approximate)
        equity_return = 0.10
        bond_return = 0.05
        cash_return = 0.025
        inflation = 0.03
        
        # Generate compound growth
        years_range = list(range(years + 1))
        equity_values = [initial_investment * (1 + equity_return) ** year for year in years_range]
        bond_values = [initial_investment * (1 + bond_return) ** year for year in years_range]
        cash_values = [initial_investment * (1 + cash_return) ** year for year in years_range]
        inflation_adjusted = [initial_investment * (1 + inflation) ** year for year in years_range]
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=years_range, y=equity_values,
            mode='lines',
            name='Equities (10% annual)',
            line=dict(color='blue', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_range, y=bond_values,
            mode='lines',
            name='Bonds (5% annual)',
            line=dict(color='green', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_range, y=cash_values,
            mode='lines',
            name='Cash (2.5% annual)',
            line=dict(color='orange', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_range, y=inflation_adjusted,
            mode='lines',
            name='Inflation (3% annual)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_returns.update_layout(
            title="Long-Term Wealth Accumulation: $10,000 Initial Investment",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            height=500,
            yaxis_type="log"
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    # Compound Returns Analysis
    compound_col1, compound_col2 = st.columns(2)
    
    with compound_col1:
        st.markdown("**Final Values After 30 Years:**")
        final_values = pd.DataFrame({
            'Asset Class': ['Equities', 'Bonds', 'Cash', 'Inflation Impact'],
            'Final Value': [f"${equity_values[-1]:,.0f}", f"${bond_values[-1]:,.0f}", 
                           f"${cash_values[-1]:,.0f}", f"${inflation_adjusted[-1]:,.0f}"],
            'Real Return': [f"${equity_values[-1]/inflation_adjusted[-1]*initial_investment:,.0f}",
                           f"${bond_values[-1]/inflation_adjusted[-1]*initial_investment:,.0f}",
                           f"${cash_values[-1]/inflation_adjusted[-1]*initial_investment:,.0f}",
                           f"${initial_investment:,.0f}"]
        })
        st.dataframe(final_values, use_container_width=True)
    
    with compound_col2:
        st.markdown("**Equity Risk Premium Analysis:**")
        
        # Calculate metrics
        equity_risk_premium = equity_return - cash_return
        bond_risk_premium = bond_return - cash_return
        
        risk_premium_data = pd.DataFrame({
            'Asset': ['Equity vs Cash', 'Bonds vs Cash', 'Equity vs Bonds'],
            'Risk Premium': [f"{equity_risk_premium:.1%}", f"{bond_risk_premium:.1%}", 
                           f"{equity_return - bond_return:.1%}"],
            'Interpretation': [
                'Compensation for volatility risk',
                'Compensation for duration risk', 
                'Additional growth premium'
            ]
        })
        st.dataframe(risk_premium_data, use_container_width=True)
    
    with st.expander("📖 Understanding the Equity Risk Premium"):
        st.markdown("""
        ### Why Equities Command a Premium
        
        The **Equity Risk Premium (ERP)** is the excess return investors demand for holding risky equities 
        over risk-free assets. Historical evidence shows this premium averages 4-6% annually over long periods.
        
        **Sources of Equity Risk Premium:**
        
        **1. Volatility Risk**
        - Equity prices fluctuate significantly in the short term
        - Investors demand compensation for bearing this uncertainty
        - Standard deviation of equity returns ~15-20% vs 3-5% for bonds
        
        **2. Business Risk**
        - Companies can fail, go bankrupt, or lose competitive advantage
        - Economic cycles affect corporate earnings
        - Technological disruption creates winners and losers
        
        **3. Liquidity Risk** 
        - While major equities are liquid, downturns can reduce market liquidity
        - Forced selling during crises can create temporary illiquidity
        - Some segments (small cap, emerging markets) have higher liquidity risk
        
        **4. Inflation Risk**
        - Equities provide better long-term inflation protection than fixed income
        - Companies can raise prices and grow earnings with inflation
        - Real assets backing equity claims maintain value
        
        ### Time Diversification Effect
        
        **Key Insight**: Equity risk decreases with longer holding periods due to:
        - Mean reversion in valuations over long periods
        - Compound growth overwhelming short-term volatility  
        - Reduced impact of market timing on final outcomes
        - Economic growth benefiting equity holders over time
        
        This explains why equity allocations typically increase with investment horizon length.
        """)
    
    # === SECTION 3: EQUITY VALUATION METHODS ===
    st.subheader("💰 How Equities Are Valued")
    
    st.markdown("""
    **Equity valuation** combines art and science, using both quantitative models and qualitative judgment 
    to estimate intrinsic value. The two primary approaches are absolute valuation (DCF models) and 
    relative valuation (multiples and comparables).
    """)
    
    # DCF Model Interactive Calculator
    with st.container():
        st.markdown("#### Discounted Cash Flow (DCF) Model")
        
        dcf_col1, dcf_col2 = st.columns(2)
        
        with dcf_col1:
            st.markdown("**DCF Model Parameters**")
            
            # DCF inputs
            year1_fcf = st.number_input("Year 1 Free Cash Flow ($M)", value=100, min_value=1)
            growth_rate = st.slider("Growth Rate (%)", 0.0, 20.0, 5.0, 1.0) / 100
            terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.5, 0.5) / 100
            discount_rate = st.slider("Discount Rate (WACC) (%)", 5.0, 15.0, 9.0, 0.5) / 100
            forecast_years = st.slider("Forecast Years", 3, 10, 5)
            
            # Generate cash flow projections
            cash_flows = []
            for year in range(forecast_years):
                fcf = year1_fcf * (1 + growth_rate) ** year
                cash_flows.append(fcf)
        
        with dcf_col2:
            st.markdown("**DCF Valuation Results**")
            
            # Calculate DCF
            dcf_results = dcf_model(cash_flows, terminal_growth, discount_rate)
            
            st.metric("Enterprise Value", f"${dcf_results['enterprise_value']:.0f}M")
            st.metric("Terminal Value", f"${dcf_results['terminal_value']:.0f}M")
            st.metric("PV of Terminal Value", f"${dcf_results['pv_terminal']:.0f}M")
            
            # Terminal value percentage
            terminal_pct = dcf_results['pv_terminal'] / dcf_results['enterprise_value']
            st.metric("Terminal Value %", f"{terminal_pct:.1%}")
    
    # Cash Flow Projection Chart
    st.markdown("#### Cash Flow Projections and Present Values")
    
    years_dcf = list(range(1, len(cash_flows) + 1))
    pv_flows = dcf_results['pv_cash_flows']
    
    fig_dcf = go.Figure()
    
    fig_dcf.add_trace(go.Bar(
        x=years_dcf, y=cash_flows,
        name='Future Cash Flows',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig_dcf.add_trace(go.Bar(
        x=years_dcf, y=pv_flows,
        name='Present Value',
        marker_color='darkblue'
    ))
    
    fig_dcf.update_layout(
        title="DCF Analysis: Future Cash Flows vs Present Values",
        xaxis_title="Year",
        yaxis_title="Cash Flow ($M)",
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig_dcf, use_container_width=True)
    
    # Valuation Multiples Section
    st.markdown("#### Common Valuation Multiples")
    
    multiples_data = pd.DataFrame({
        'Multiple': ['P/E Ratio', 'EV/EBITDA', 'P/B Ratio', 'PEG Ratio', 'P/S Ratio'],
        'Formula': [
            'Price / Earnings Per Share',
            'Enterprise Value / EBITDA', 
            'Price / Book Value Per Share',
            'P/E Ratio / Growth Rate',
            'Price / Sales Per Share'
        ],
        'What It Measures': [
            'Price relative to earnings',
            'Valuation relative to operating cash flow',
            'Price relative to accounting book value',
            'P/E adjusted for growth expectations', 
            'Valuation relative to revenue'
        ],
        'Typical Range': [
            '12-25x',
            '8-15x',
            '1-4x',
            '0.5-2.0x',
            '1-6x'
        ],
        'Best For': [
            'Mature, profitable companies',
            'Cross-industry comparisons',
            'Asset-heavy businesses',
            'Growth stocks',
            'Early-stage/loss-making companies'
        ]
    })
    
    st.dataframe(multiples_data, use_container_width=True)
    
    with st.expander("📖 Understanding Valuation Methods"):
        st.markdown("""
        ### DCF Model: The Fundamental Approach
        
        **Discounted Cash Flow** models attempt to estimate intrinsic value by projecting future cash flows 
        and discounting them to present value using the company's cost of capital.
        
        **DCF Formula:**
        ```
        Enterprise Value = Σ[FCF_t / (1 + WACC)^t] + Terminal Value / (1 + WACC)^n
        ```
        
        **Key Components:**
        - **Free Cash Flow**: Cash available to all capital providers after investments
        - **WACC**: Weighted Average Cost of Capital (discount rate)
        - **Terminal Value**: Value beyond explicit forecast period
        - **Growth Assumptions**: Revenue, margin, and investment projections
        
        **DCF Strengths:**
        - Fundamental approach based on business value creation
        - Forces detailed analysis of business drivers
        - Less influenced by market sentiment
        - Theoretically sound valuation foundation
        
        **DCF Limitations:**
        - Highly sensitive to assumptions (garbage in, garbage out)
        - Terminal value often dominates (60-80% of total value)
        - Difficult for cyclical or rapidly changing businesses
        - Long-term forecasts inherently uncertain
        
        ### Multiples: Relative Valuation
        
        **Valuation multiples** compare a company's price to various financial metrics, 
        providing relative valuation context within peer groups or historical ranges.
        
        **Multiple Selection Guidelines:**
        
        **P/E Ratio**: Best for stable, profitable companies
        - Forward P/E uses next year's expected earnings
        - Trailing P/E uses last 12 months actual earnings
        - Cyclically adjusted P/E (CAPE) smooths earnings over 10 years
        
        **EV/EBITDA**: Excellent for cross-industry comparison
        - Excludes capital structure differences
        - EBITDA proxy for operating cash flow
        - Useful for leveraged or capital-intensive businesses
        
        **PEG Ratio**: Adjusts P/E for growth expectations
        - PEG < 1.0 suggests undervaluation relative to growth
        - PEG > 2.0 may indicate overvaluation
        - Most useful for growth stocks with predictable earnings
        
        **Critical Insight**: No single valuation method is perfect. Professional analysts typically 
        use multiple approaches and triangulate to a valuation range rather than a precise price target.
        """)
    
    # === SECTION 4: PORTFOLIO CONSTRUCTION ROLE ===
    st.subheader("📊 Equities in Portfolio Construction")
    
    portfolio_tabs = st.tabs(["🎯 Strategic Role", "🌍 Diversification", "⚖️ Factor Investing"])
    
    with portfolio_tabs[0]:
        st.markdown("### Core Role in Multi-Asset Portfolios")
        
        # Risk-Return by Asset Class
        asset_classes = ['Cash', '10Y Bonds', 'IG Corporate', 'REITs', 'Commodities', 'Intl Developed', 'US Large Cap', 'US Small Cap', 'Emerging Markets']
        expected_returns = [2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 10.0, 10.5]
        volatilities = [1.0, 8.0, 6.0, 18.0, 20.0, 16.0, 15.0, 20.0, 25.0]
        
        fig_asset_classes = go.Figure()
        
        colors = ['gray', 'green', 'lightgreen', 'brown', 'orange', 'blue', 'darkblue', 'purple', 'red']
        
        for i, (asset, ret, vol) in enumerate(zip(asset_classes, expected_returns, volatilities)):
            fig_asset_classes.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode='markers+text',
                name=asset,
                text=[asset],
                textposition="top center",
                marker=dict(size=12, color=colors[i]),
                showlegend=False
            ))
        
        fig_asset_classes.update_layout(
            title="Risk-Return Profile: Asset Classes",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            height=500
        )
        
        st.plotly_chart(fig_asset_classes, use_container_width=True)
        
        st.markdown("""
        **Strategic Allocation Principles:**
        
        - **Growth Engine**: Equities typically form the growth component of portfolios
        - **Risk-Return Trade-off**: Higher expected returns come with higher volatility
        - **Time Horizon Matching**: Longer horizons can tolerate higher equity allocations
        - **Diversification Benefits**: Combining asset classes can improve risk-adjusted returns
        """)
    
    with portfolio_tabs[1]:
        st.markdown("### Diversification: Sector, Geography, and Size")
        
        # Sector Allocation
        sectors, weights = create_sector_allocation_data()
        
        fig_sectors = go.Figure(data=[go.Pie(
            labels=sectors, 
            values=weights,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig_sectors.update_layout(
            title="S&P 500 Sector Allocation (Approximate)",
            height=400
        )
        
        st.plotly_chart(fig_sectors, use_container_width=True)
        
        diversification_principles = pd.DataFrame({
            'Diversification Type': ['Sector', 'Geographic', 'Market Cap', 'Style'],
            'Purpose': [
                'Reduce industry-specific risks',
                'Reduce country/currency risks',
                'Capture size premiums', 
                'Balance growth vs value exposure'
            ],
            'Implementation': [
                'Spread across 8-11 sectors',
                'US, International Developed, EM',
                'Large, mid, small cap allocation',
                'Growth and value factor exposure'
            ],
            'Key Risk': [
                'Sector rotation cycles',
                'Currency and political risk',
                'Size factor timing risk',
                'Style factor cyclicality'
            ]
        })
        
        st.dataframe(diversification_principles, use_container_width=True)
    
    with portfolio_tabs[2]:
        st.markdown("### Factor Investing: Beyond Market Beta")
        
        st.markdown("""
        **Factor investing** seeks to capture systematic risk premiums beyond broad market exposure 
        by tilting portfolios toward specific characteristics that historically generate excess returns.
        """)
        
        # Factor Performance Chart
        dates, factor_returns = calculate_factor_returns()
        
        fig_factors = go.Figure()
        
        colors_factors = ['black', 'blue', 'green', 'purple', 'orange', 'red']
        for i, (factor, returns) in enumerate(factor_returns.items()):
            fig_factors.add_trace(go.Scatter(
                x=dates, y=returns * 100,
                mode='lines',
                name=factor,
                line=dict(color=colors_factors[i], width=2)
            ))
        
        fig_factors.update_layout(
            title="Factor Performance Over Time (Cumulative Returns)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=500
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # Factor Description Table
        factor_descriptions = pd.DataFrame({
            'Factor': ['Market (Beta)', 'Value', 'Growth', 'Quality', 'Low Volatility', 'Momentum'],
            'Definition': [
                'Broad market exposure',
                'Cheap stocks (low P/E, P/B)', 
                'High growth expectations',
                'Strong balance sheets, profitability',
                'Lower risk, stable earnings',
                'Recent price performance'
            ],
            'Academic Evidence': [
                'CAPM - market risk premium',
                'Fama-French - value premium',
                'Growth stocks outperform in some periods', 
                'Quality firms have lower risk',
                'Low-vol stocks have better risk-adj returns',
                'Momentum persists 3-12 months'
            ],
            'Implementation': [
                'Broad market index funds',
                'Value index/ETFs, fundamental indexing',
                'Growth index/ETFs, high P/E screens',
                'Quality screens, ESG factors',
                'Low-vol index funds, min-variance',
                'Momentum index funds, trend following'
            ]
        })
        
        st.dataframe(factor_descriptions, use_container_width=True)
    
    with st.expander("📖 Understanding Factor Investing"):
        st.markdown("""
        ### The Evolution from CAPM to Multi-Factor Models
        
        **Capital Asset Pricing Model (CAPM)** originally suggested that beta (market sensitivity) 
        was the only factor needed to explain expected returns:
        
        ```
        Expected Return = Risk-Free Rate + Beta × Market Risk Premium
        ```
        
        However, empirical research revealed additional factors that systematically drive returns.

        ### Fama-French Three-Factor Model (1993)
        
        **Enhanced the CAPM** by adding two additional factors:
        
        ```
        R = Rf + β(Rm - Rf) + s×SMB + h×HML
        ```
        
        Where:
        - **SMB**: Small Minus Big (size factor)
        - **HML**: High Minus Low (value factor)
        
        ### Modern Factor Zoo
        
        **Research has identified 300+ potential factors**, though many lack statistical significance 
        or economic intuition. The most robust factors with persistent evidence include:
        
        **1. Market (Beta)**: Systematic risk exposure - higher beta stocks move more with markets
        
        **2. Size**: Small-cap stocks historically outperform large-cap (though inconsistent recently)
        
        **3. Value**: Cheap stocks (low P/E, P/B) outperform expensive stocks over long periods
        
        **4. Momentum**: Stocks with recent strong performance continue outperforming (3-12 months)
        
        **5. Quality**: Companies with strong balance sheets, stable earnings show better risk-adjusted returns
        
        **6. Low Volatility**: Counterintuitively, low-risk stocks often outperform high-risk stocks
        
        ### Factor Implementation Considerations
        
        **Factor Timing**: Different factors perform well in different market regimes
        - Value performs better during market stress and rising rates
        - Growth/Momentum excel during expansions and falling rates  
        - Quality provides defensive characteristics during uncertainty
        
        **Factor Concentration**: Avoid over-tilting portfolios to single factors
        - Diversify across multiple factors to reduce timing risk
        - Monitor factor loadings and correlations over time
        - Consider factor cyclicality in long-term allocations
        """)
    
    # === SECTION 5: MARKET DYNAMICS AND VALUATION CYCLES ===
    st.subheader("📈 Market Dynamics and Valuation Cycles")
    
    st.markdown("""
    **Equity markets** don't move in straight lines - they exhibit cyclical behavior driven by economic cycles, 
    sentiment shifts, and valuation mean reversion. Understanding these patterns helps in long-term investment planning.
    """)
    
    # P/E Ratio Historical Analysis
    with st.container():
        st.markdown("#### Historical P/E Ratios: Valuation Cycles")
        
        pe_dates, pe_ratios = calculate_pe_ratio_history()
        
        fig_pe = go.Figure()
        
        # P/E ratio line
        fig_pe.add_trace(go.Scatter(
            x=pe_dates, y=pe_ratios,
            mode='lines',
            name='P/E Ratio',
            line=dict(color='blue', width=2)
        ))
        
        # Add mean line
        pe_mean = np.mean(pe_ratios)
        fig_pe.add_hline(
            y=pe_mean, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Long-term Average: {pe_mean:.1f}x"
        )
        
        # Add bands
        pe_std = np.std(pe_ratios)
        fig_pe.add_hrect(
            y0=pe_mean - pe_std, y1=pe_mean + pe_std,
            fillcolor="lightgray", opacity=0.2,
            annotation_text="±1 Standard Deviation"
        )
        
        fig_pe.update_layout(
            title="S&P 500 P/E Ratio: 25-Year History (Simulated)",
            xaxis_title="Year",
            yaxis_title="Price-to-Earnings Ratio",
            height=500
        )
        
        st.plotly_chart(fig_pe, use_container_width=True)
        
        st.markdown("""
        **This chart demonstrates several key valuation concepts:**
        
        - **Mean Reversion**: P/E ratios tend to revert to long-term averages over time
        - **Cyclical Nature**: Valuations move in multi-year cycles tied to economic conditions  
        - **Crisis Periods**: Major disruptions (2008 financial crisis, 2020 pandemic) create valuation extremes
        - **Opportunity Identification**: Extreme readings often signal potential turning points
        
        **Investment Implications:**
        - High P/E periods often precede lower future returns
        - Low P/E periods historically offer better long-term entry points
        - Dollar-cost averaging helps smooth valuation timing issues
        """)
    
    # Interest Rates and Equity Valuation
    with st.container():
        st.markdown("#### Interest Rates and Equity Valuation")
        
        # Create interest rate sensitivity analysis
        rates = np.arange(0.02, 0.12, 0.01)
        base_earnings = 100
        base_multiple = 20
        
        # Simplified relationship: lower rates = higher multiples
        multiples = []
        for rate in rates:
            # Inverse relationship with interest rates (simplified)
            multiple = base_multiple * (0.08 / rate) ** 0.5  # Square root relationship
            multiples.append(multiple)
        
        valuations = [base_earnings * mult for mult in multiples]
        
        fig_rates = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_rates.add_trace(
            go.Scatter(x=rates*100, y=multiples, name="P/E Multiple", line=dict(color="blue")),
            secondary_y=False,
        )
        
        fig_rates.add_trace(
            go.Scatter(x=rates*100, y=valuations, name="Stock Price", line=dict(color="red")),
            secondary_y=True,
        )
        
        fig_rates.update_xaxes(title_text="Interest Rate (%)")
        fig_rates.update_yaxes(title_text="P/E Multiple", secondary_y=False)
        fig_rates.update_yaxes(title_text="Stock Price ($)", secondary_y=True)
        fig_rates.update_layout(title="Interest Rate Sensitivity: The 'Equity Duration' Effect")
        
        st.plotly_chart(fig_rates, use_container_width=True)
        
        st.markdown("""
        **This relationship illustrates why equities have "duration-like" characteristics:**
        
        - **Lower Interest Rates**: Increase present value of future cash flows → Higher valuations
        - **Higher Interest Rates**: Reduce present value calculations → Lower valuations  
        - **Growth Stocks**: More sensitive to rate changes (longer "duration")
        - **Value Stocks**: Less sensitive due to nearer-term cash flows
        
        **The "Fed Put" Phenomenon**: Markets often rally when central banks cut rates, 
        as lower discount rates mechanically increase equity valuations.
        """)
    
    # === SECTION 6: BEHAVIORAL AND SENTIMENT FACTORS ===
    st.subheader("🧠 Behavioral Finance and Market Psychology")
    
    st.markdown("""
    **Traditional finance theory** assumes rational actors and efficient markets, but real-world equity markets 
    are significantly influenced by psychological biases, sentiment cycles, and behavioral patterns.
    """)
    
    behavior_tabs = st.tabs(["🎭 Key Biases", "📊 Sentiment Indicators", "🔄 Market Cycles"])
    
    with behavior_tabs[0]:
        st.markdown("### Cognitive Biases in Equity Investing")
        
        bias_data = pd.DataFrame({
            'Bias': ['Loss Aversion', 'Anchoring', 'Confirmation Bias', 'Overconfidence', 'Herding', 'Recency Bias'],
            'Description': [
                'Losses feel twice as painful as equivalent gains',
                'Over-relying on first piece of information received',
                'Seeking info that confirms existing beliefs',
                'Overestimating one\'s investment abilities',  
                'Following crowd behavior, especially in extremes',
                'Overweighting recent events in decision-making'
            ],
            'Market Impact': [
                'Reluctance to sell losing positions',
                'Stock prices "stick" to historical levels',
                'Ignoring contrary evidence about holdings',
                'Excessive trading, inadequate diversification',
                'Bubble formation and crash amplification', 
                'Overreacting to recent news/performance'
            ],
            'Mitigation Strategy': [
                'Set stop-losses, systematic rebalancing',
                'Use multiple valuation anchors',
                'Actively seek contrary viewpoints',
                'Track performance, use systematic approaches',
                'Independent research, contrarian thinking',
                'Focus on long-term fundamentals'
            ]
        })
        
        st.dataframe(bias_data, use_container_width=True)
        
    with behavior_tabs[1]:
        st.markdown("### Market Sentiment Indicators")
        
        # Create sentiment indicator simulation
        sentiment_dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        np.random.seed(42)
        
        # VIX simulation (fear gauge)
        base_vix = 20
        vix_values = []
        market_returns = []
        
        for i, date in enumerate(sentiment_dates):
            # Create cyclical volatility with crisis spikes
            cycle = 5 * np.sin(2 * np.pi * i / 24)  # 2-year cycle
            noise = np.random.normal(0, 3)
            
            # COVID spike
            if date.year == 2020 and 3 <= date.month <= 6:
                spike = 40
            else:
                spike = 0
                
            vix = max(10, base_vix + cycle + noise + spike)
            vix_values.append(vix)
            
            # Inverse relationship with returns (simplified)
            monthly_return = (30 - vix) / 100 + np.random.normal(0, 0.02)
            market_returns.append(monthly_return)
        
        fig_sentiment = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_sentiment.add_trace(
            go.Scatter(x=sentiment_dates, y=vix_values, name="VIX (Fear Index)", line=dict(color="red")),
            secondary_y=False,
        )
        
        cumulative_returns = np.cumprod(1 + np.array(market_returns)) - 1
        fig_sentiment.add_trace(
            go.Scatter(x=sentiment_dates, y=cumulative_returns*100, name="Market Returns (%)", line=dict(color="blue")),
            secondary_y=True,
        )
        
        fig_sentiment.update_xaxes(title_text="Date")
        fig_sentiment.update_yaxes(title_text="VIX Level", secondary_y=False)
        fig_sentiment.update_yaxes(title_text="Cumulative Return (%)", secondary_y=True)
        fig_sentiment.update_layout(title="Fear & Greed: VIX vs Market Performance")
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        sentiment_indicators = pd.DataFrame({
            'Indicator': ['VIX (Volatility Index)', 'Put/Call Ratio', 'Margin Debt', 'IPO Activity', 'Insider Trading'],
            'What It Measures': [
                'Market fear/complacency via option prices',
                'Bearish vs bullish options positioning',
                'Investor leverage and risk appetite',
                'New issue demand and speculation',
                'Corporate insider buying vs selling'
            ],
            'Bullish Signal': ['VIX > 30 (extreme fear)', 'Ratio > 1.2 (high puts)', 'Declining margin debt', 'Low IPO volume', 'Heavy insider buying'],
            'Bearish Signal': ['VIX < 15 (complacency)', 'Ratio < 0.8 (low puts)', 'Rising margin debt', 'IPO boom/frenzy', 'Heavy insider selling'],
            'Reliability': ['High (contrarian)', 'Medium', 'Medium-High', 'Medium', 'High (fundamental)']
        })
        
        st.dataframe(sentiment_indicators, use_container_width=True)
    
    with behavior_tabs[2]:
        st.markdown("### Market Cycle Psychology")
        
        # Create market cycle visualization
        cycle_stages = ['Pessimism', 'Hope', 'Optimism', 'Excitement', 'Euphoria', 'Anxiety', 'Denial', 'Fear', 'Desperation', 'Panic']
        cycle_prices = [20, 30, 45, 65, 100, 85, 70, 50, 35, 15]
        cycle_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'yellow', 'orange', 'red', 'darkred', 'black']
        
        fig_cycle = go.Figure()
        
        # Create cycle path
        angles = np.linspace(0, 2*np.pi, len(cycle_stages), endpoint=False)
        x_coords = np.cos(angles) * np.array(cycle_prices)
        y_coords = np.sin(angles) * np.array(cycle_prices)
        
        fig_cycle.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines+markers+text',
            text=cycle_stages,
            textposition="middle center",
            marker=dict(size=15, color=cycle_colors),
            line=dict(width=3, color='blue'),
            showlegend=False
        ))
        
        fig_cycle.update_layout(
            title="The Psychology of Market Cycles",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_cycle, use_container_width=True)
        
        st.markdown("""
        **Key Insights from Market Cycle Psychology:**
        
        - **Peak Emotions = Poor Timing**: Maximum bullishness occurs at tops, maximum bearishness at bottoms
        - **Contrarian Opportunities**: Best investment opportunities often feel most uncomfortable
        - **Media Amplification**: News coverage tends to be most positive at peaks, most negative at troughs
        - **Time Arbitrage**: Patient investors can profit from others' emotional decision-making
        
        **Practical Applications:**
        - Use systematic rebalancing to counteract emotional decision-making
        - Increase buying during periods of high pessimism (high VIX, negative sentiment)
        - Reduce risk exposure during periods of excessive optimism and speculation
        - Focus on long-term fundamentals rather than short-term market movements
        """)
    
    # === SECTION 7: ESG AND MODERN CONSIDERATIONS ===
    st.subheader("🌱 ESG Integration and Modern Equity Considerations")
    
    st.markdown("""
    **Environmental, Social, and Governance (ESG)** factors have become increasingly important in equity analysis 
    and portfolio construction, reflecting both investor preferences and materiality to long-term returns.
    """)
    
    esg_tabs = st.tabs(["🌍 ESG Integration", "📊 Performance Impact", "🔮 Future Trends"])
    
    with esg_tabs[0]:
        st.markdown("### ESG Integration in Equity Analysis")
        
        esg_framework = pd.DataFrame({
            'ESG Category': ['Environmental', 'Social', 'Governance'],
            'Key Factors': [
                'Carbon emissions, resource use, waste management, climate risk',
                'Labor practices, diversity, community relations, product safety', 
                'Board structure, executive compensation, shareholder rights, ethics'
            ],
            'Financial Materiality': [
                'Regulatory risk, operational efficiency, long-term sustainability',
                'Employee retention, brand reputation, regulatory compliance',
                'Decision-making quality, risk management, capital allocation'
            ],
            'Investment Integration': [
                'Carbon intensity screening, green revenue exposure',
                'Workplace diversity metrics, customer satisfaction scores',
                'Board independence, management quality assessment'
            ]
        })
        
        st.dataframe(esg_framework, use_container_width=True)
        
        st.markdown("""
        **ESG Integration Approaches:**
        
        **1. Negative Screening**: Exclude sectors/companies based on ESG criteria
        - Traditional approach: tobacco, weapons, gambling exclusions
        - Modern screening: fossil fuels, poor governance, social violations
        
        **2. Positive Screening**: Select companies with strong ESG characteristics  
        - Best-in-class selection within sectors
        - ESG leaders across portfolio construction
        
        **3. ESG Integration**: Incorporate ESG factors into fundamental analysis
        - Adjust discount rates for ESG risks
        - Include ESG metrics in valuation models
        - Consider long-term sustainability of business models
        
        **4. Impact Investing**: Target measurable positive outcomes
        - Direct environmental or social benefits
        - Thematic investing (clean energy, healthcare access)
        - Shareholder engagement for corporate change
        """)
    
    with esg_tabs[1]:
        st.markdown("### ESG Performance and Risk Characteristics")
        
        # Create ESG performance simulation
        performance_dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')
        np.random.seed(42)
        
        # Simulate ESG vs traditional performance
        esg_returns = np.random.normal(0.008, 0.035, len(performance_dates))  # Slightly lower vol
        traditional_returns = np.random.normal(0.0075, 0.040, len(performance_dates))  # Higher vol
        
        esg_cumulative = np.cumprod(1 + esg_returns) - 1
        traditional_cumulative = np.cumprod(1 + traditional_returns) - 1
        
        fig_esg = go.Figure()
        
        fig_esg.add_trace(go.Scatter(
            x=performance_dates, y=esg_cumulative * 100,
            mode='lines',
            name='ESG Portfolio',
            line=dict(color='green', width=2)
        ))
        
        fig_esg.add_trace(go.Scatter(
            x=performance_dates, y=traditional_cumulative * 100,
            mode='lines', 
            name='Traditional Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        fig_esg.update_layout(
            title="ESG vs Traditional Equity Performance (Simulated)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        st.plotly_chart(fig_esg, use_container_width=True)
        
        # ESG Performance metrics
        esg_metrics = pd.DataFrame({
            'Metric': ['Annualized Return', 'Volatility', 'Sharpe Ratio', 'Maximum Drawdown', 'ESG Score'],
            'ESG Portfolio': ['9.8%', '14.2%', '0.69', '-18.5%', '8.2/10'],
            'Traditional Portfolio': ['9.1%', '16.8%', '0.54', '-22.1%', '5.4/10'],
            'Difference': ['+0.7%', '-2.6%', '+0.15', '+3.6%', '+2.8']
        })
        
        st.dataframe(esg_metrics, use_container_width=True)
    
    with esg_tabs[2]:
        st.markdown("### Future Trends in Equity Markets")
        
        future_trends = pd.DataFrame({
            'Trend': ['Climate Disclosure', 'AI & Technology', 'Demographic Shifts', 'Geopolitical Fragmentation'],
            'Impact on Equities': [
                'Mandatory climate risk reporting affecting valuations',
                'AI disruption creating winners/losers across sectors',
                'Aging populations affecting consumption patterns', 
                'Supply chain resilience becoming valuation factor'
            ],
            'Investment Implications': [
                'Carbon-intensive industries face valuation discounts',
                'Tech disruption accelerates, traditional moats erode',
                'Healthcare, leisure sectors benefit; traditional retail struggles',
                'Geographic diversification becomes more complex'
            ],
            'Timeline': ['2025-2030', '2024-2035', '2025-2050', '2024-2030']
        })
        
        st.dataframe(future_trends, use_container_width=True)
        
        st.markdown("""
        **Key Considerations for Future Equity Investing:**
        
        **1. Regulatory Evolution**: ESG reporting becoming mandatory globally
        - EU Taxonomy requirements affecting capital flows
        - SEC climate disclosure rules in development  
        - Carbon pricing expanding globally
        
        **2. Technological Disruption**: AI and automation reshaping industries
        - Labor-intensive sectors facing margin pressure
        - Data and network effects creating new competitive advantages
        - Traditional valuation models may need updating
        
        **3. Demographic Tailwinds/Headwinds**: 
        - Aging developed markets creating healthcare opportunities
        - Younger emerging market populations driving consumption
        - Workforce participation changes affecting productivity
        
        **4. Sustainability as Competitive Advantage**:
        - Resource efficiency becoming cost advantage
        - Brand differentiation through sustainability
        - Access to capital increasingly tied to ESG performance
        """)
    
    # === SECTION 8: MARKET STRUCTURE AND LIQUIDITY ===
    st.subheader("🏗️ Market Structure and Modern Trading Environment")
    
    st.markdown("""
    **Understanding equity market structure** is crucial for investors, as the mechanics of how stocks trade 
    can significantly impact execution quality, liquidity, and price discovery.
    """)
    
    market_structure_tabs = st.tabs(["🏢 Trading Venues", "🤖 Market Participants", "💧 Liquidity Dynamics"])
    
    with market_structure_tabs[0]:
        st.markdown("### Modern Equity Trading Ecosystem")
        
        venue_data = pd.DataFrame({
            'Venue Type': ['NYSE (Specialist)', 'NASDAQ (Market Maker)', 'Dark Pools', 'Electronic ECNs', 'Alternative Trading Systems'],
            'Market Share': ['~25%', '~20%', '~15%', '~25%', '~15%'],
            'Key Characteristics': [
                'Auction-based, designated market makers',
                'Multiple market makers, electronic matching',
                'Hidden liquidity, institutional focus',
                'Fully electronic, order matching',
                'Alternative execution venues'
            ],
            'Advantages': [
                'Price discovery, transparency',
                'Competition among market makers',
                'Reduced market impact for large orders',
                'Fast execution, low costs',
                'Specialized execution strategies'
            ],
            'Considerations': [
                'Potential for slower execution',
                'Spread competition benefits',
                'Less transparent pricing',
                'May fragment liquidity',
                'Regulatory complexity'
            ]
        })
        
        st.dataframe(venue_data, use_container_width=True)
        
        st.markdown("""
        **Market Fragmentation Impact:**
        
        - **Positive**: Competition among venues reduces trading costs
        - **Negative**: Liquidity spread across multiple venues can reduce efficiency
        - **Regulation NMS**: Ensures best execution across all venues
        - **Smart Order Routing**: Technology routes orders to best available prices
        """)
    
    with market_structure_tabs[1]:
        st.markdown("### Key Market Participants and Their Roles")
        
        # Visualize market participant ecosystem
        participants = ['Retail Investors', 'Institutional Investors', 'High-Frequency Traders', 'Market Makers', 'Pension Funds', 'Hedge Funds']
        volumes = [15, 30, 25, 20, 25, 35]  # Approximate daily volume impact
        holding_periods = [365, 180, 0.1, 1, 1800, 90]  # Days
        
        fig_participants = go.Figure()
        
        colors_participants = ['lightblue', 'blue', 'red', 'green', 'purple', 'orange']
        for i, (participant, volume, holding) in enumerate(zip(participants, volumes, holding_periods)):
            fig_participants.add_trace(go.Scatter(
                x=[holding], y=[volume],
                mode='markers+text',
                name=participant,
                text=[participant],
                textposition="top center",
                marker=dict(size=20, color=colors_participants[i], opacity=0.7),
                showlegend=False
            ))
        
        fig_participants.update_layout(
            title="Market Participants: Volume Impact vs Holding Period",
            xaxis_title="Average Holding Period (Days, Log Scale)",
            xaxis_type="log",
            yaxis_title="Daily Volume Impact (%)",
            height=500
        )
        
        st.plotly_chart(fig_participants, use_container_width=True)
        
        participant_roles = pd.DataFrame({
            'Participant': ['Retail Investors', 'Institutional Investors', 'High-Frequency Traders', 'Market Makers'],
            'Primary Role': [
                'Long-term wealth building',
                'Professional asset management',
                'Short-term arbitrage/market making',
                'Provide liquidity and price discovery'
            ],
            'Trading Behavior': [
                'Buy-and-hold, emotion-driven',
                'Fundamental analysis, large blocks',
                'Algorithm-driven, millisecond decisions',
                'Bid-offer spreads, inventory management'
            ],
            'Market Impact': [
                'Trend following, momentum creation',
                'Price stability, fundamental anchoring',
                'Reduced spreads, increased volatility',
                'Improved liquidity, tighter spreads'
            ]
        })
        
        st.dataframe(participant_roles, use_container_width=True)
    
    with market_structure_tabs[2]:
        st.markdown("### Liquidity Dynamics and Market Stress")
        
        # Simulate liquidity conditions
        stress_dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Simulate bid-ask spreads
        base_spread = 0.05  # 5 basis points
        spreads = []
        
        for date in stress_dates:
            # Normal conditions
            normal_noise = np.random.normal(0, 0.01)
            
            # Crisis conditions (COVID)
            if date.year == 2020 and 3 <= date.month <= 5:
                crisis_mult = 5
            else:
                crisis_mult = 1
                
            spread = base_spread * crisis_mult + normal_noise
            spreads.append(max(0.01, spread))
        
        # Create liquidity visualization
        fig_liquidity = go.Figure()
        
        fig_liquidity.add_trace(go.Scatter(
            x=stress_dates, y=np.array(spreads) * 10000,  # Convert to basis points
            mode='lines',
            name='Bid-Ask Spread',
            line=dict(color='red', width=1)
        ))
        
        fig_liquidity.update_layout(
            title="Market Liquidity: Bid-Ask Spreads During Stress (Simulated)",
            xaxis_title="Date",
            yaxis_title="Bid-Ask Spread (Basis Points)",
            height=400
        )
        
        st.plotly_chart(fig_liquidity, use_container_width=True)
        
        st.markdown("""
        **Understanding Liquidity Risk:**
        
        - **Normal Markets**: Tight spreads (1-5 basis points for large caps)
        - **Stress Periods**: Spreads can widen dramatically (20+ basis points)
        - **Market Impact**: Large orders move prices more during illiquid periods
        - **Flight to Quality**: Large-cap, high-volume stocks maintain better liquidity
        
        **Liquidity Management Strategies:**
        - **Time Diversification**: Spread large trades over time
        - **Dark Pools**: Hide order information to reduce market impact
        - **Algorithm Trading**: TWAP/VWAP strategies for optimal execution
        - **Liquidity Buffers**: Maintain cash reserves for opportunities/obligations
        """)
    
    # === SECTION 9: ADVANCED TOPICS AND RISKS ===
    st.subheader("⚠️ Advanced Considerations and Risk Management")
    
    advanced_tabs = st.tabs(["📉 Tail Risks", "🔄 Correlation Dynamics", "💱 Currency Exposure"])
    
    with advanced_tabs[0]:
        st.markdown("### Tail Risk and Black Swan Events")
        
        # Simulate return distribution with fat tails
        np.random.seed(42)
        normal_returns = np.random.normal(0.001, 0.015, 5000)  # Daily returns
        
        # Add fat tails (occasional extreme moves)
        extreme_events = np.random.choice([0, 1], size=5000, p=[0.99, 0.01])
        extreme_magnitude = np.random.normal(0, 0.08, 5000) * extreme_events
        actual_returns = normal_returns + extreme_magnitude
        
        # Create distribution comparison
        fig_tails = go.Figure()
        
        fig_tails.add_trace(go.Histogram(
            x=normal_returns * 100,
            name='Normal Distribution',
            opacity=0.7,
            nbinsx=50,
            histnorm='probability density'
        ))
        
        fig_tails.add_trace(go.Histogram(
            x=actual_returns * 100,
            name='Actual Market Returns (Fat Tails)',
            opacity=0.7,
            nbinsx=50,
            histnorm='probability density'
        ))
        
        fig_tails.update_layout(
            title="Return Distributions: Normal vs Fat-Tailed",
            xaxis_title="Daily Return (%)",
            yaxis_title="Probability Density",
            height=400,
            barmode='overlay'
        )
        
        st.plotly_chart(fig_tails, use_container_width=True)
        
        tail_risk_events = pd.DataFrame({
            'Event Type': ['Market Crash', 'Flash Crash', 'Currency Crisis', 'Geopolitical Shock', 'Pandemic'],
            'Historical Examples': [
                '1987, 2008, 2020',
                '2010 Flash Crash',
                'Asian Crisis 1997, Brexit',
                '9/11, Gulf Wars',
                'COVID-19, SARS'
            ],
            'Typical Magnitude': ['-20% to -50%', '-5% to -10% (minutes)', '-10% to -30%', '-5% to -15%', '-30% to -40%'],
            'Duration': ['Months to years', 'Minutes to hours', 'Months', 'Days to weeks', 'Months'],
            'Mitigation': [
                'Diversification, hedging',
                'Circuit breakers, position limits',
                'Geographic diversification',
                'Scenario planning, cash reserves',
                'Sector diversification, defensive assets'
            ]
        })
        
        st.dataframe(tail_risk_events, use_container_width=True)
    
    with advanced_tabs[1]:
        st.markdown("### Correlation Dynamics in Crisis vs Normal Times")
        
        # Simulate correlation changes
        correlation_dates = pd.date_range(start='2008-01-01', end='2024-12-31', freq='M')
        
        normal_correlation = 0.6
        crisis_correlation = 0.9
        correlations = []
        
        for date in correlation_dates:
            # Crisis periods
            if (date.year == 2008 and date.month >= 9) or (date.year == 2009) or (date.year == 2020 and 3 <= date.month <= 6):
                base_corr = crisis_correlation
            else:
                base_corr = normal_correlation
                
            # Add some noise
            corr = base_corr + np.random.normal(0, 0.05)
            correlations.append(max(0.2, min(0.95, corr)))
        
        fig_corr = go.Figure()
        
        fig_corr.add_trace(go.Scatter(
            x=correlation_dates, y=correlations,
            mode='lines',
            name='Inter-Stock Correlation',
            line=dict(color='purple', width=2)
        ))
        
        fig_corr.add_hline(
            y=normal_correlation, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Normal Times: {normal_correlation:.1f}"
        )
        
        fig_corr.add_hline(
            y=crisis_correlation, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Crisis Periods: {crisis_correlation:.1f}"
        )
        
        fig_corr.update_layout(
            title="Correlation Breakdown: When Diversification Fails",
            xaxis_title="Year",
            yaxis_title="Average Stock Correlation",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        **The Correlation Problem:**
        
        - **Normal Times**: Stocks move somewhat independently (correlation ~0.6)
        - **Crisis Periods**: Almost all stocks fall together (correlation >0.9)
        - **"When you need diversification most, it works least"**
        - **Sector diversification becomes less effective during systemic stress**
        
        **Implications for Portfolio Construction:**
        - Geographic and asset class diversification more important than stock picking
        - Consider alternative assets (bonds, commodities, real estate) for true diversification
        - Maintain adequate cash reserves for crisis opportunities
        - Use options or other hedging strategies for tail risk protection
        """)
    
    with advanced_tabs[2]:
        st.markdown("### Currency Exposure in International Equities")
        
        # Simulate currency impact on international returns
        fx_dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        np.random.seed(42)
        
        # Simulate local vs USD returns
        local_returns = np.random.normal(0.008, 0.04, len(fx_dates))  # 0.8% monthly in local currency
        fx_returns = np.random.normal(0, 0.03, len(fx_dates))  # Currency volatility
        
        local_cumulative = np.cumprod(1 + local_returns) - 1
        usd_cumulative = np.cumprod(1 + local_returns + fx_returns) - 1
        
        fig_fx = go.Figure()
        
        fig_fx.add_trace(go.Scatter(
            x=fx_dates, y=local_cumulative * 100,
            mode='lines',
            name='Local Currency Returns',
            line=dict(color='blue', width=2)
        ))
        
        fig_fx.add_trace(go.Scatter(
            x=fx_dates, y=usd_cumulative * 100,
            mode='lines',
            name='USD Returns (with FX impact)',
            line=dict(color='red', width=2)
        ))
        
        fig_fx.update_layout(
            title="Currency Impact on International Equity Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        st.plotly_chart(fig_fx, use_container_width=True)
        
        currency_considerations = pd.DataFrame({
            'Aspect': ['Return Impact', 'Volatility Impact', 'Hedging Options', 'Cost Considerations'],
            'USD Strengthening': [
                'Reduces international returns',
                'Adds volatility to returns',
                'Currency hedged funds available',
                'Hedging costs 0.2-0.8% annually'
            ],
            'USD Weakening': [
                'Enhances international returns',
                'Can smooth overall portfolio returns',
                'May choose unhedged exposure',
                'Opportunity cost of hedging'
            ],
            'Strategic Considerations': [
                'Long-term USD trends matter most',
                'Short-term volatility vs long-term trends',
                'Hedge ratio decisions (0%, 50%, 100%)',
                'Balance hedging costs vs risk reduction'
            ]
        })
        
        st.dataframe(currency_considerations, use_container_width=True)
        
        st.markdown("""
        **Currency Hedging Decision Framework:**
        
        **Arguments for Hedging:**
        - Reduces portfolio volatility
        - Focuses returns on underlying equity performance  
        - Eliminates currency timing risk
        - More predictable outcomes for planning
        
        **Arguments Against Hedging:**
        - Costs money (hedge premium)
        - Currency can provide diversification benefit
        - USD weakness can enhance international returns
        - Adds complexity to investment process
        
        **Common Approaches:**
        - **Full Hedge (100%)**: Maximum volatility reduction
        - **Partial Hedge (50%)**: Balance of risk reduction and cost
        - **No Hedge (0%)**: Accept full currency exposure
        - **Dynamic Hedging**: Adjust based on market conditions
        """)
    
    # === SECTION 10: CONCLUSION AND KEY TAKEAWAYS ===
    st.subheader("🎯 Key Takeaways: Building an Equity Investment Framework")
    
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; color: white;">
            <h4>Essential Principles for Equity Investing</h4>
        </div>
        """, unsafe_allow_html=True)
        
        conclusion_cols = st.columns(2)
        
        with conclusion_cols[0]:
            st.markdown("""
            **Fundamental Understanding:**
            - Equities represent residual ownership with unlimited upside potential
            - Long-term wealth creation through compounding and economic growth  
            - Valuation combines quantitative models with qualitative judgment
            - Factor exposure beyond market beta can enhance returns
            
            **Risk Management:**
            - Diversification across sectors, geographies, and market caps
            - Time horizon matching with volatility tolerance
            - Understanding behavioral biases and market psychology
            - Systematic approach to reduce emotional decision-making
            """)
        
        with conclusion_cols[1]:
            st.markdown("""
            **Portfolio Construction:**
            - Strategic asset allocation based on long-term objectives
            - Factor diversification to capture multiple risk premiums  
            - ESG integration for risk management and values alignment
            - Regular rebalancing to maintain target exposures
            
            **Modern Considerations:**
            - Technology disruption accelerating industry transformation
            - Climate risk becoming material valuation factor
            - Demographic trends creating sector rotation opportunities  
            - Geopolitical fragmentation affecting global diversification
            """)
    
    # Advanced Risk Management Summary
    with st.container():
        st.markdown("#### Advanced Risk Management Checklist")
        
        risk_checklist = pd.DataFrame({
            'Risk Category': ['Valuation Risk', 'Liquidity Risk', 'Concentration Risk', 'Currency Risk', 'Tail Risk'],
            'Key Indicators': [
                'P/E ratios, cyclical valuations',
                'Bid-ask spreads, market depth',
                'Single stock/sector weights',
                'Unhedged international exposure',
                'Correlation spikes, VIX levels'
            ],
            'Monitoring Tools': [
                'Historical P/E charts, sector rotation',
                'Average daily volume, market impact',
                'Position sizing rules, diversification metrics',
                'Currency hedging ratios',
                'Scenario analysis, stress testing'
            ],
            'Mitigation Strategies': [
                'Dollar-cost averaging, contrarian positioning',
                'Position limits, time diversification',
                'Maximum position sizes, sector caps',
                'Hedging decisions, natural hedges',
                'Tail hedges, defensive allocations'
            ]
        })
        
        st.dataframe(risk_checklist, use_container_width=True)
    
    # Interactive Summary Tool
    with st.expander("🧮 Personal Equity Allocation Calculator", expanded=False):
        st.markdown("### Determine Your Equity Allocation")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            age = st.slider("Age", 20, 80, 35)
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            time_horizon = st.slider("Investment Horizon (years)", 1, 40, 10)
            income_stability = st.selectbox("Income Stability", ["Stable", "Variable", "Uncertain"])
        
        with calc_col2:
            # Calculate suggested allocation
            base_equity = 100 - age  # Age-based rule
            
            # Risk tolerance adjustment
            risk_adj = {"Conservative": -10, "Moderate": 0, "Aggressive": +10}[risk_tolerance]
            
            # Time horizon adjustment  
            horizon_adj = min(10, time_horizon - 10)
            
            # Income stability adjustment
            income_adj = {"Stable": +5, "Variable": 0, "Uncertain": -10}[income_stability]
            
            suggested_equity = max(20, min(90, base_equity + risk_adj + horizon_adj + income_adj))
            
            st.metric("Suggested Equity Allocation", f"{suggested_equity}%")
            st.metric("Suggested Bond Allocation", f"{100-suggested_equity}%")
            
            st.markdown(f"""
            **Allocation Rationale:**
            - Base allocation (100 - age): {100-age}%
            - Risk tolerance adjustment: {risk_adj:+d}%
            - Time horizon adjustment: {horizon_adj:+d}%  
            - Income stability adjustment: {income_adj:+d}%
            
            **Final Allocation: {suggested_equity}% Equity / {100-suggested_equity}% Bonds**
            
            *Note: This is a simplified calculator. Consult with a financial advisor for personalized advice.*
            """)
    
    # Final Implementation Framework
    with st.expander("📋 Equity Investment Implementation Framework", expanded=False):
        st.markdown("### Step-by-Step Implementation Guide")
        
        implementation_steps = pd.DataFrame({
            'Step': ['1. Define Objectives', '2. Risk Assessment', '3. Asset Allocation', '4. Security Selection', '5. Implementation', '6. Monitoring'],
            'Key Actions': [
                'Set time horizon, return targets, constraints',
                'Evaluate risk tolerance, liquidity needs',
                'Determine equity vs bond allocation',
                'Choose index funds vs active vs individual stocks',
                'Execute trades, set up automatic investing',
                'Regular rebalancing, performance review'
            ],
            'Considerations': [
                'Be realistic about returns and volatility',
                'Consider worst-case scenarios',
                'Start conservative, increase over time',
                'Cost-conscious, diversified approach',
                'Dollar-cost averaging for large amounts',
                'Quarterly review, annual rebalancing'
            ],
            'Common Mistakes': [
                'Unrealistic expectations, lack of planning',
                'Overconfidence, ignoring downside risk',
                'Too aggressive initially, panic selling',
                'Over-diversification, high fees',
                'Market timing, emotional decisions',
                'Neglect, over-trading, tax inefficiency'
            ]
        })
        
        st.dataframe(implementation_steps, use_container_width=True)
        
        st.markdown("""
        **Final Wisdom for Equity Investors:**
        
        ✅ **Do:**
        - Start early, invest regularly, stay disciplined
        - Focus on low costs and broad diversification
        - Understand what you own and why you own it
        - Plan for both bull and bear markets
        - Rebalance systematically, not emotionally
        
        ❌ **Don't:**
        - Try to time the market or chase performance
        - Put all eggs in one basket (sector/stock/geography)
        - Make investment decisions based on news or emotions
        - Neglect to consider taxes and fees
        - Abandon your long-term plan during short-term volatility
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
        <p><strong>📚 Congratulations on completing this comprehensive equity fundamentals guide!</strong></p>
        <p>Remember: successful equity investing is a marathon, not a sprint. Focus on time-tested principles, 
        continuous learning, and disciplined execution rather than trying to outsmart the market.</p>
        <p><em>"The stock market is a device for transferring money from the impatient to the patient." - Warren Buffett</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_equity_fundamentals_page()
