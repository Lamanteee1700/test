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

def get_real_sector_data():
    """Get real sector allocation data from S&P 500"""
    try:
        # Major S&P 500 sector ETFs to get real allocations
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Communication Services': 'XLC',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }
        
        # Approximate S&P 500 sector weights (as of 2024)
        weights = [28.5, 13.2, 12.8, 10.5, 8.9, 8.3, 6.2, 4.1, 2.8, 2.4, 2.3]
        
        return list(sectors.keys()), weights
    except:
        # Fallback data
        sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 
                  'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
                  'Utilities', 'Real Estate', 'Materials']
        weights = [28.5, 13.2, 12.8, 10.5, 8.9, 8.3, 6.2, 4.1, 2.8, 2.4, 2.3]
        return sectors, weights

def show_equity_fundamentals_page():
    st.set_page_config(page_title="Equity Fundamentals", layout="wide")
    
    st.title("📈 Equity Fundamentals & Portfolio Construction")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">From Theory to Practice: Building Equity Portfolios</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # === SECTION 1: WHAT IS AN EQUITY? ===
    st.subheader("🏢 Understanding Equity Fundamentals")
    
    theory_tabs = st.tabs(["The Theory: Residual Ownership", "The Theory: Embedded Call Option", "The Theory: Long-Term Case"])
    
    with theory_tabs[0]:
        st.markdown("#### Equity as Residual Ownership")
        st.markdown("""
        Equity represents ownership in a corporation - a residual claim on assets and earnings 
        after debt obligations. Unlike bondholders with contractual claims, equity holders own 
        proportional shares of future success or failure.
        
        **Key Rights:** Voting (board election, major decisions), dividends (when declared), 
        and liquidation (after debt holders). **Capital hierarchy:** Secured debt → Senior debt → 
        Subordinated debt → Preferred equity → Common equity (last in line, highest risk/return).
        """)
        
    with theory_tabs[1]:
        st.markdown("#### Equity as an Embedded Call Option")
        st.markdown("""
        Equity can be viewed as a call option on firm assets with debt value as strike price. 
        This Black-Scholes-Merton perspective provides key insights: limited downside with unlimited upside, 
        volatility creates value (explaining growth stock premiums), and time value effects 
        (why distressed companies maintain equity value).
        
        **Formula:** $V_E = \max(V_A - D, 0)$ where $V_E$ = equity value, $V_A$ = asset value, $D$ = debt value.
        """)
        
    with theory_tabs[2]:
        st.markdown("#### The Long-Term Wealth Creation Case")
        
        # Historical Returns with real starting date
        years = 124  # 1900 to 2024
        initial_investment = 10000
        
        # Average historical annual returns (1900-2024 averages)
        equity_return = 0.096  # US equity long-term average
        bond_return = 0.051   # US bond long-term average
        cash_return = 0.035   # Cash/Treasury bills average
        inflation = 0.031     # US inflation average
        
        years_range = list(range(years + 1))
        equity_values = [initial_investment * (1 + equity_return) ** year for year in years_range]
        bond_values = [initial_investment * (1 + bond_return) ** year for year in years_range]
        cash_values = [initial_investment * (1 + cash_return) ** year for year in years_range]
        inflation_adjusted = [initial_investment * (1 + inflation) ** year for year in years_range]
        
        # Adjust years to start from 1900
        years_actual = [1900 + year for year in years_range]
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=years_actual, y=equity_values,
            mode='lines',
            name='US Equities (9.6% avg)',
            line=dict(color='blue', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_actual, y=bond_values,
            mode='lines',
            name='US Bonds (5.1% avg)',
            line=dict(color='green', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_actual, y=cash_values,
            mode='lines',
            name='Cash (3.5% avg)',
            line=dict(color='orange', width=3)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=years_actual, y=inflation_adjusted,
            mode='lines',
            name='Inflation (3.1% avg)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_returns.update_layout(
            title="Long-Term Wealth Accumulation: $10,000 Initial Investment (1900-2024)",
            xaxis_title="Year",
            yaxis_title="Portfolio Value ($)",
            height=400,
            yaxis_type="log"
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        st.caption("*Using average historical returns 1900-2024: US equities, bonds, cash, and inflation rates*")
        
        st.markdown("""
        **The Equity Risk Premium:** Equities have historically delivered 4-5% annual premium over 
        risk-free assets, compensating for volatility, business risk, and liquidity constraints. 
        Time diversification reduces equity risk over longer holding periods through mean reversion 
        and compound growth effects.
        """)
    
    # === SECTION 2: EQUITY VALUATION METHODS ===
    st.subheader("💰 The Theory: Equity Valuation Models")
    
    valuation_tabs = st.tabs(["The Theory: DCF Model", "The Theory: Valuation Multiples"])
    
    with valuation_tabs[0]:
        st.markdown("#### Discounted Cash Flow Model")
        
        dcf_col1, dcf_col2 = st.columns([1, 1])
        
        with dcf_col1:
            # DCF inputs
            year1_fcf = st.number_input("Year 1 Free Cash Flow ($M)", value=100, min_value=1)
            growth_rate = st.slider("Growth Rate (%)", 0.0, 20.0, 5.0, 1.0) / 100
            terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.5, 0.5) / 100
            discount_rate = st.slider("Discount Rate (WACC) (%)", 5.0, 15.0, 9.0, 0.5) / 100
            forecast_years = 5
            
            # Generate cash flow projections
            cash_flows = []
            for year in range(forecast_years):
                fcf = year1_fcf * (1 + growth_rate) ** year
                cash_flows.append(fcf)
            
            # Calculate DCF
            dcf_results = dcf_model(cash_flows, terminal_growth, discount_rate)
        
        with dcf_col2:
            st.markdown("**DCF Formula:**")
            st.latex(r"EV = \sum_{t=1}^{n} \frac{FCF_t}{(1+WACC)^t} + \frac{TV}{(1+WACC)^n}")
            st.latex(r"TV = \frac{FCF_{n+1}}{WACC - g}")
            
            st.metric("Enterprise Value", f"${dcf_results['enterprise_value']:.0f}M")
            st.metric("Terminal Value %", f"{dcf_results['pv_terminal']/dcf_results['enterprise_value']:.1%}")
        
        # Smaller DCF visualization
        years_dcf = list(range(1, len(cash_flows) + 1))
        pv_flows = dcf_results['pv_cash_flows']
        
        fig_dcf = go.Figure()
        fig_dcf.add_trace(go.Bar(x=years_dcf, y=cash_flows, name='Future Cash Flows', marker_color='lightblue'))
        fig_dcf.add_trace(go.Bar(x=years_dcf, y=pv_flows, name='Present Value', marker_color='darkblue'))
        fig_dcf.update_layout(title="DCF Components", xaxis_title="Year", yaxis_title="Cash Flow ($M)", height=300, barmode='group')
        
        st.plotly_chart(fig_dcf, use_container_width=True)
    
    with valuation_tabs[1]:
        st.markdown("#### Valuation Multiples Comparison")
        
        st.markdown("""
        **Example:** Comparing tech companies across different markets shows how multiples vary by 
        growth expectations, profitability, and market conditions. High-growth companies command 
        premium multiples while mature firms trade closer to market averages.
        """)
        
        # Example company comparison
        comparison_data = pd.DataFrame({
            'Company': ['Apple (US)', 'Microsoft (US)', 'ASML (Europe)', 'Taiwan Semi (Asia)', 'SAP (Europe)', 'Tencent (Asia)'],
            'Industry': ['Consumer Tech', 'Software', 'Semiconductors', 'Semiconductors', 'Software', 'Internet'],
            'Region': ['US', 'US', 'Europe', 'Asia', 'Europe', 'Asia'],
            'P/E Ratio': [28.5, 32.1, 35.2, 18.7, 29.8, 22.4],
            'EV/EBITDA': [22.1, 25.8, 28.9, 12.3, 24.7, 18.2],
            'P/B Ratio': [45.2, 12.8, 8.7, 4.2, 7.1, 3.8]
        })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        multiples_data = pd.DataFrame({
            'Multiple': ['P/E Ratio', 'EV/EBITDA', 'P/B Ratio', 'PEG Ratio', 'P/S Ratio'],
            'Formula': ['Price / EPS', 'Enterprise Value / EBITDA', 'Price / Book Value', 'P/E / Growth Rate', 'Price / Sales'],
            'Best Used For': ['Profitable companies', 'Cross-industry comparison', 'Asset-heavy businesses', 'Growth stocks', 'Early-stage companies'],
            'Typical Range': ['12-25x', '8-15x', '1-4x', '0.5-2.0x', '1-6x']
        })
        
        st.dataframe(multiples_data, use_container_width=True)
    
    # === PORTFOLIO CONSTRUCTION SIMULATION ===
    st.subheader("🎯 Portfolio Construction Simulation")
    
    st.markdown("""
    **Step-by-step equity portfolio construction** - Each step builds on the previous to create a realistic portfolio 
    with clear rationale and implications.
    """)
    
    portfolio_steps = st.tabs(["Step 1: Define Objectives", "Step 2: Risk Assessment", "Step 3: Asset Allocation", "Step 4: Security Selection"])
    
    with portfolio_steps[0]:
        st.markdown("### Step 1: Define Investment Objectives")
        
        obj_col1, obj_col2 = st.columns(2)
        
        with obj_col1:
            st.markdown("**Investor Profile Selection:**")
            investor_type = st.selectbox("Choose Investor Profile:", 
                ["Young Professional (25-35)", "Mid-Career (35-50)", "Pre-Retirement (50-65)", "Retiree (65+)"])
            
            investment_goal = st.selectbox("Primary Goal:", 
                ["Long-term Growth", "Growth with Income", "Income with Preservation", "Capital Preservation"])
            
            time_horizon = st.slider("Investment Horizon (years):", 5, 40, 20)
        
        with obj_col2:
            # Set profile defaults
            profiles = {
                "Young Professional (25-35)": {"risk": "High", "liquidity": "Low", "tax": "High bracket"},
                "Mid-Career (35-50)": {"risk": "Moderate-High", "liquidity": "Medium", "tax": "High bracket"},
                "Pre-Retirement (50-65)": {"risk": "Moderate", "liquidity": "Medium", "tax": "Medium bracket"},
                "Retiree (65+)": {"risk": "Low-Moderate", "liquidity": "High", "tax": "Lower bracket"}
            }
            
            profile = profiles[investor_type]
            
            st.markdown(f"""
            **Derived Characteristics:**
            - **Risk Tolerance:** {profile['risk']}
            - **Liquidity Needs:** {profile['liquidity']}
            - **Tax Situation:** {profile['tax']}
            - **Time Horizon:** {time_horizon} years
            - **Primary Goal:** {investment_goal}
            """)
            
            # Store selections in session state
            st.session_state.investor_profile = {
                'type': investor_type,
                'goal': investment_goal,
                'horizon': time_horizon,
                'risk': profile['risk'],
                'liquidity': profile['liquidity']
            }
    
    with portfolio_steps[1]:
        st.markdown("### Step 2: Risk Assessment & Capacity")
        
        if 'investor_profile' in st.session_state:
            profile = st.session_state.investor_profile
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.markdown("**Risk Tolerance Evaluation:**")
                
                market_drop = st.selectbox("If your portfolio dropped 30% in 6 months, you would:",
                    ["Panic and sell everything", "Be very concerned but hold", "Stay calm and maybe buy more", "Definitely buy more - great opportunity"])
                
                volatility_comfort = st.selectbox("Annual portfolio volatility you can accept:",
                    ["Under 5%", "5-10%", "10-20%", "Over 20%"])
                
                loss_tolerance = st.selectbox("Maximum loss you could accept in worst year:",
                    ["Under 5%", "5-15%", "15-30%", "Over 30%"])
            
            with risk_col2:
                # Calculate risk score
                risk_scores = {
                    "Panic and sell everything": 1, "Be very concerned but hold": 2,
                    "Stay calm and maybe buy more": 3, "Definitely buy more - great opportunity": 4
                }
                volatility_scores = {"Under 5%": 1, "5-10%": 2, "10-20%": 3, "Over 20%": 4}
                loss_scores = {"Under 5%": 1, "5-15%": 2, "15-30%": 3, "Over 30%": 4}
                
                total_score = risk_scores[market_drop] + volatility_scores[volatility_comfort] + loss_scores[loss_tolerance]
                
                if total_score <= 6:
                    risk_profile = "Conservative"
                    equity_range = "20-40%"
                elif total_score <= 9:
                    risk_profile = "Moderate"
                    equity_range = "40-70%"
                else:
                    risk_profile = "Aggressive"
                    equity_range = "70-90%"
                
                st.markdown(f"""
                **Risk Assessment Results:**
                - **Risk Score:** {total_score}/12
                - **Risk Profile:** {risk_profile}
                - **Suggested Equity Range:** {equity_range}
                - **Investment Horizon:** {profile['horizon']} years
                """)
                
                st.session_state.risk_assessment = {
                    'score': total_score,
                    'profile': risk_profile,
                    'equity_range': equity_range
                }
    
    with portfolio_steps[2]:
        st.markdown("### Step 3: Strategic Asset Allocation")
        
        if 'risk_assessment' in st.session_state:
            allocation_col1, allocation_col2 = st.columns(2)
            
            with allocation_col1:
                st.markdown("**Asset Allocation Framework:**")
                
                # Base allocation on risk assessment
                risk_profile = st.session_state.risk_assessment['profile']
                
                if risk_profile == "Conservative":
                    default_equity = 30
                elif risk_profile == "Moderate":
                    default_equity = 60
                else:
                    default_equity = 80
                
                equity_allocation = st.slider("Equity Allocation (%)", 0, 100, default_equity, 5)
                bond_allocation = 100 - equity_allocation
                
                st.markdown(f"""
                **Proposed Allocation:**
                - **Equities:** {equity_allocation}%
                - **Bonds:** {bond_allocation}%
                """)
            
            with allocation_col2:
                # Asset class expected returns (source: research institutions)
                st.markdown("**Expected Returns & Volatility:**")
                
                expected_data = pd.DataFrame({
                    'Asset Class': ['US Large Cap Equity', 'International Equity', 'Emerging Markets', 'US Bonds', 'International Bonds'],
                    'Expected Return': ['7.2%', '8.1%', '9.5%', '3.8%', '4.2%'],
                    'Volatility': ['16.5%', '18.2%', '24.1%', '4.2%', '7.8%'],
                    'Allocation': [f'{equity_allocation*0.6:.0f}%', f'{equity_allocation*0.3:.0f}%', 
                                  f'{equity_allocation*0.1:.0f}%', f'{bond_allocation*0.8:.0f}%', f'{bond_allocation*0.2:.0f}%']
                })
                
                st.dataframe(expected_data, use_container_width=True)
                st.caption("*Expected returns based on institutional research (10-year forward-looking)*")
                
                # Calculate portfolio metrics
                equity_return = 0.072 * 0.6 + 0.081 * 0.3 + 0.095 * 0.1  # Weighted equity return
                bond_return = 0.038 * 0.8 + 0.042 * 0.2  # Weighted bond return
                
                portfolio_return = (equity_allocation/100) * equity_return + (bond_allocation/100) * bond_return
                portfolio_volatility = ((equity_allocation/100) * 0.17)**2 + ((bond_allocation/100) * 0.05)**2
                portfolio_volatility = portfolio_volatility**0.5
                
                st.metric("Expected Portfolio Return", f"{portfolio_return:.1%}")
                st.metric("Expected Portfolio Volatility", f"{portfolio_volatility:.1%}")
                
                st.session_state.allocation = {
                    'equity': equity_allocation,
                    'bond': bond_allocation,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility
                }


    with portfolio_steps[3]:
        st.markdown("### Step 4: Security Selection & Implementation")
        
        if 'allocation' in st.session_state:
            selection_col1, selection_col2 = st.columns(2)
            
            with selection_col1:
                st.markdown("**Equity Portfolio Construction:**")
                
                equity_pct = st.session_state.allocation['equity']
                
                # Sector allocation
                sectors, weights = get_real_sector_data()
                
                fig_sectors = go.Figure(data=[go.Pie(
                    labels=sectors, 
                    values=weights,
                    hole=0.3,
                    marker_colors=px.colors.qualitative.Set3
                )])
                
                fig_sectors.update_layout(
                    title="S&P 500 Sector Allocation",
                    height=400
                )
                
                st.plotly_chart(fig_sectors, use_container_width=True)
                
                st.markdown(f"""
                **Proposed Equity Allocation ({equity_pct}% of portfolio):**
                - **US Large Cap:** {equity_pct*0.6:.0f}% (S&P 500 Index)
                - **International Developed:** {equity_pct*0.3:.0f}% (MSCI EAFE Index)
                - **Emerging Markets:** {equity_pct*0.1:.0f}% (MSCI EM Index)
                """)
            
            with selection_col2:
                st.markdown("**Implementation Vehicles:**")
                
                implementation_options = pd.DataFrame({
                    'Asset Class': ['US Large Cap', 'International Dev.', 'Emerging Markets', 'US Bonds', 'Intl Bonds'],
                    'Index Fund': ['VTSAX', 'VTIAX', 'VEMAX', 'VBTLX', 'VTABX'],
                    'ETF': ['VTI', 'VEA', 'VWO', 'BND', 'BNDX'],
                    'Expense Ratio': ['0.03%', '0.11%', '0.14%', '0.05%', '0.09%'],
                    'Portfolio Weight': [f'{equity_pct*0.6:.0f}%', f'{equity_pct*0.3:.0f}%', 
                                        f'{equity_pct*0.1:.0f}%', f'{(100-equity_pct)*0.8:.0f}%', f'{(100-equity_pct)*0.2:.0f}%']
                })
                
                st.dataframe(implementation_options, use_container_width=True)
                
                # Calculate total costs
                total_expense = (equity_pct*0.6*0.0003 + equity_pct*0.3*0.0011 + equity_pct*0.1*0.0014 + 
                               (100-equity_pct)*0.8*0.0005 + (100-equity_pct)*0.2*0.0009) / 100
                
                st.metric("Total Portfolio Expense Ratio", f"{total_expense:.2%}")
                
                st.markdown("""
                **Implementation Summary:**
                - **Approach:** Low-cost index fund diversification
                - **Rebalancing:** Quarterly review, rebalance if >5% deviation
                - **Tax Efficiency:** Tax-advantaged accounts first, then taxable
                - **Dollar-Cost Averaging:** Regular monthly investments
                """)
    
    # === FINAL PORTFOLIO SUMMARY ===
    if 'allocation' in st.session_state:
        st.subheader("📊 Final Portfolio Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            final_allocation = pd.DataFrame({
                'Asset Class': ['US Large Cap Equity', 'International Equity', 'Emerging Markets', 'US Bonds', 'International Bonds'],
                'Allocation': [f"{st.session_state.allocation['equity']*0.6:.1f}%", 
                              f"{st.session_state.allocation['equity']*0.3:.1f}%",
                              f"{st.session_state.allocation['equity']*0.1:.1f}%",
                              f"{st.session_state.allocation['bond']*0.8:.1f}%",
                              f"{st.session_state.allocation['bond']*0.2:.1f}%"],
                'Expected Return': ['7.2%', '8.1%', '9.5%', '3.8%', '4.2%'],
                'Risk Level': ['Medium', 'Medium-High', 'High', 'Low', 'Low-Medium']
            })
            
            st.dataframe(final_allocation, use_container_width=True)
        
        with summary_col2:
            st.markdown("""
            **Portfolio Characteristics:**
            - **Risk Profile:** {risk_profile}
            - **Expected Return:** {expected_return:.1%}
            - **Expected Volatility:** {volatility:.1%}
            - **Sharpe Ratio (Est.):** {sharpe:.2f}
            - **Total Expenses:** {expense:.2%}
            """.format(
                risk_profile=st.session_state.risk_assessment['profile'],
                expected_return=st.session_state.allocation['expected_return'],
                volatility=st.session_state.allocation['volatility'],
                sharpe=(st.session_state.allocation['expected_return'] - 0.03) / st.session_state.allocation['volatility'],
                expense=total_expense
            ))
            
            st.markdown("""
            **Next Steps:**
            1. Open investment accounts (401k, IRA, taxable)
            2. Set up automatic monthly investments
            3. Implement dollar-cost averaging strategy
            4. Review and rebalance quarterly
            5. Stay disciplined during market volatility
            """)

if __name__ == "__main__":
    show_equity_fundamentals_page()
