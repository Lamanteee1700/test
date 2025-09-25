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
    
    # === INSTITUTIONAL PORTFOLIO CONSTRUCTION SIMULATION ===
    st.subheader("🎯 Institutional Portfolio Construction Simulation")
    
    st.markdown("""
    **Step-by-step institutional equity portfolio construction** - Each step demonstrates how institutional investors 
    build large-scale equity portfolios with sophisticated risk management and implementation strategies.
    """)
    
    portfolio_steps = st.tabs(["Step 1: Define Mandate", "Step 2: Risk Budget", "Step 3: Strategic Allocation", "Step 4: Implementation"])
    
    with portfolio_steps[0]:
        st.markdown("### Step 1: Define Investment Mandate")
        
        obj_col1, obj_col2 = st.columns(2)
        
        with obj_col1:
            st.markdown("**Institution Type Selection:**")
            institution_type = st.selectbox("Choose Institution Type:", 
                ["Corporate Pension Fund", "Public Pension Fund", "University Endowment", "Insurance Company", "Sovereign Wealth Fund"])
            
            mandate_objective = st.selectbox("Primary Mandate:", 
                ["Growth + Income", "Liability Matching", "Absolute Return", "Capital Preservation", "Intergenerational Equity"])
            
            aum_size = st.selectbox("Assets Under Management:", 
                ["$1-5 Billion", "$5-20 Billion", "$20-100 Billion", "$100B+"])
            
            time_horizon = st.selectbox("Investment Horizon:", 
                ["10-15 years", "15-25 years", "25+ years (perpetual)", "Variable (liability-driven)"])
        
        with obj_col2:
            # Set institutional profile defaults
            institutional_profiles = {
                "Corporate Pension Fund": {"risk_tolerance": "Moderate", "liquidity": "Medium", "governance": "Corporate Board", "constraint": "ERISA compliance"},
                "Public Pension Fund": {"risk_tolerance": "Moderate-High", "liquidity": "Medium", "governance": "Board of Trustees", "constraint": "Political oversight"},
                "University Endowment": {"risk_tolerance": "High", "liquidity": "Low-Medium", "governance": "Investment Committee", "constraint": "Spending rate targets"},
                "Insurance Company": {"risk_tolerance": "Low-Moderate", "liquidity": "High", "governance": "Risk Committee", "constraint": "Regulatory capital"},
                "Sovereign Wealth Fund": {"risk_tolerance": "High", "liquidity": "Low", "governance": "Government Board", "constraint": "Political/Strategic"}
            }
            
            profile = institutional_profiles[institution_type]
            
            st.markdown(f"""
            **Institutional Characteristics:**
            - **Risk Tolerance:** {profile['risk_tolerance']}
            - **Liquidity Requirements:** {profile['liquidity']}
            - **Governance Structure:** {profile['governance']}
            - **Key Constraint:** {profile['constraint']}
            - **Time Horizon:** {time_horizon}
            - **AUM Size:** {aum_size}
            """)
            
            # Store selections in session state
            st.session_state.institution_profile = {
                'type': institution_type,
                'mandate': mandate_objective,
                'aum': aum_size,
                'horizon': time_horizon,
                'risk_tolerance': profile['risk_tolerance'],
                'liquidity': profile['liquidity'],
                'governance': profile['governance']
            }
    
    with portfolio_steps[1]:
        st.markdown("### Step 2: Risk Budget & Constraints")
        
        if 'institution_profile' in st.session_state:
            profile = st.session_state.institution_profile
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.markdown("**Risk Budget Framework:**")
                
                tracking_error = st.selectbox("Target Tracking Error vs Benchmark:",
                    ["1-2% (Conservative)", "2-4% (Moderate)", "4-6% (Active)", "6%+ (Aggressive)"])
                
                var_limit = st.selectbox("Value-at-Risk (95% confidence, monthly):",
                    ["2-3%", "3-5%", "5-8%", "8%+"])
                
                concentration_limit = st.selectbox("Maximum Single Position:",
                    ["2%", "3%", "5%", "No limit"])
                
                sector_limit = st.selectbox("Maximum Sector Concentration:",
                    ["Index weight + 3%", "Index weight + 5%", "Index weight + 8%", "No sector limits"])
            
            with risk_col2:
                # Calculate institutional risk profile
                te_scores = {"1-2% (Conservative)": 1, "2-4% (Moderate)": 2, "4-6% (Active)": 3, "6%+ (Aggressive)": 4}
                var_scores = {"2-3%": 1, "3-5%": 2, "5-8%": 3, "8%+": 4}
                conc_scores = {"2%": 1, "3%": 2, "5%": 3, "No limit": 4}
                
                risk_score = te_scores[tracking_error] + var_scores[var_limit] + conc_scores[concentration_limit]
                
                if risk_score <= 6:
                    strategy_type = "Core/Passive"
                    equity_target = "45-55%"
                elif risk_score <= 9:
                    strategy_type = "Core-Plus/Enhanced"  
                    equity_target = "50-70%"
                else:
                    strategy_type = "Active/Satellite"
                    equity_target = "60-80%"
                
                # Regulatory constraints based on institution type
                if profile['type'] == "Insurance Company":
                    regulatory_note = "Solvency II capital requirements limit equity allocation"
                elif profile['type'] == "Corporate Pension Fund":
                    regulatory_note = "ERISA fiduciary standards require prudent diversification"  
                elif profile['type'] == "Public Pension Fund":
                    regulatory_note = "State regulations may limit alternative investments"
                else:
                    regulatory_note = "Minimal regulatory constraints"
                
                st.markdown(f"""
                **Risk Assessment Results:**
                - **Risk Budget Score:** {risk_score}/12
                - **Strategy Classification:** {strategy_type}
                - **Target Equity Range:** {equity_target}
                - **Tracking Error Budget:** {tracking_error}
                - **Regulatory Constraint:** {regulatory_note}
                """)
                
                st.session_state.risk_assessment = {
                    'score': risk_score,
                    'strategy': strategy_type,
                    'equity_target': equity_target,
                    'tracking_error': tracking_error,
                    'var_limit': var_limit
                }


    with portfolio_steps[2]:
        st.markdown("### Step 3: Strategic Asset Allocation")
        
        if 'risk_assessment' in st.session_state:
            allocation_col1, allocation_col2 = st.columns(2)
            
            with allocation_col1:
                st.markdown("**Strategic Asset Allocation Framework:**")
                
                # Base allocation on institutional risk profile
                strategy_type = st.session_state.risk_assessment['strategy']
                
                if strategy_type == "Core/Passive":
                    default_equity = 50
                elif strategy_type == "Core-Plus/Enhanced":
                    default_equity = 60
                else:
                    default_equity = 70
                
                equity_allocation = st.slider("Public Equity Allocation (%)", 30, 80, default_equity, 5)
                
                # Institutional-specific allocations
                private_equity = st.slider("Private Equity (%)", 0, 25, 10)
                real_estate = st.slider("Real Estate (%)", 0, 15, 8)
                fixed_income = st.slider("Fixed Income (%)", 10, 40, 25)
                alternatives = 100 - equity_allocation - private_equity - real_estate - fixed_income
                
                st.markdown(f"""
                **Institutional Asset Allocation:**
                - **Public Equity:** {equity_allocation}%
                - **Private Equity:** {private_equity}%
                - **Real Estate:** {real_estate}%
                - **Fixed Income:** {fixed_income}%
                - **Alternatives/Cash:** {alternatives}%
                """)
            
            with allocation_col2:
                # Institutional-grade expected returns and risk metrics
                st.markdown("**Expected Returns & Risk (10-Year Forward):**")
                
                institutional_data = pd.DataFrame({
                    'Asset Class': ['Public Equity', 'Private Equity', 'Real Estate', 'Fixed Income', 'Alternatives'],
                    'Expected Return': ['7.5%', '9.2%', '6.8%', '4.2%', '5.5%'],
                    'Volatility': ['16.8%', '22.4%', '14.2%', '4.8%', '12.1%'],
                    'Liquidity': ['Daily', 'Quarterly', 'Monthly', 'Daily', 'Variable'],
                    'Allocation': [f'{equity_allocation}%', f'{private_equity}%', f'{real_estate}%', 
                                  f'{fixed_income}%', f'{alternatives}%']
                })
                
                st.dataframe(institutional_data, use_container_width=True)
                st.caption("*Expected returns based on institutional investment consultant forecasts*")
                
                # Calculate institutional portfolio metrics
                weights = [equity_allocation/100, private_equity/100, real_estate/100, fixed_income/100, alternatives/100]
                returns = [0.075, 0.092, 0.068, 0.042, 0.055]
                volatilities = [0.168, 0.224, 0.142, 0.048, 0.121]
                
                portfolio_return = sum(w * r for w, r in zip(weights, returns))
                # Simplified portfolio volatility calculation
                portfolio_volatility = sum(w * v for w, v in zip(weights, volatilities)) * 0.8  # Diversification benefit
                
                st.metric("Expected Portfolio Return", f"{portfolio_return:.1%}")
                st.metric("Expected Portfolio Volatility", f"{portfolio_volatility:.1%}")
                st.metric("Expected Sharpe Ratio", f"{(portfolio_return - 0.035)/portfolio_volatility:.2f}")
                
                st.session_state.allocation = {
                    'public_equity': equity_allocation,
                    'private_equity': private_equity,
                    'real_estate': real_estate,
                    'fixed_income': fixed_income,
                    'alternatives': alternatives,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility
                }
    
    with portfolio_steps[3]:
        st.markdown("### Step 4: Implementation Strategy")
        
        if 'allocation' in st.session_state:
            impl_col1, impl_col2 = st.columns(2)
            
            with impl_col1:
                st.markdown("**Public Equity Implementation:**")
                
                equity_pct = st.session_state.allocation['public_equity']
                
                # Advanced institutional equity implementation
                st.markdown(f"""
                **Core-Satellite Structure ({equity_pct}% of total portfolio):**
                - **Core Holdings (70%):** {equity_pct*0.7:.0f}%
                  - Global Index: 40% 
                  - Regional Indices: 30%
                - **Satellite Holdings (30%):** {equity_pct*0.3:.0f}%
                  - Active Managers: 15%
                  - Factor Tilts: 10%
                  - Tactical Allocation: 5%
                """)
                
                # Sector allocation visualization
                sectors, weights = get_real_sector_data()
                
                fig_sectors = go.Figure(data=[go.Pie(
                    labels=sectors, 
                    values=weights,
                    hole=0.3,
                    marker_colors=px.colors.qualitative.Set3
                )])
                
                fig_sectors.update_layout(
                    title="Core Holdings: Global Equity Allocation",
                    height=350
                )
                
                st.plotly_chart(fig_sectors, use_container_width=True)
            
            with impl_col2:
                st.markdown("**Institutional Implementation Vehicles:**")
                
                implementation_options = pd.DataFrame({
                    'Strategy': ['Passive Core', 'Enhanced Index', 'Active Long-Only', 'Long-Short Equity', 'Private Equity'],
                    'Allocation': [f'{equity_pct*0.4:.0f}%', f'{equity_pct*0.2:.0f}%', f'{equity_pct*0.15:.0f}%', 
                                  f'{equity_pct*0.15:.0f}%', f'{st.session_state.allocation["private_equity"]:.0f}%'],
                    'Expected Alpha': ['0%', '0.5-1%', '1-3%', '2-5%', '3-6%'],
                    'Fee Range': ['0.05-0.15%', '0.15-0.35%', '0.75-1.25%', '1.5-2.5%', '2.0-3.0% + 20%'],
                    'Risk Budget': ['Tracking error <1%', 'TE 1-2%', 'TE 3-5%', 'TE 5-10%', 'Illiquidity premium']
                })
                
                st.dataframe(implementation_options, use_container_width=True)
                
                st.markdown("**Operational Considerations:**")
                
                operational_framework = pd.DataFrame({
                    'Area': ['Trading', 'Risk Management', 'Performance', 'Governance'],
                    'Approach': [
                        'Multi-manager platform, transition management',
                        'Real-time monitoring, stress testing, VaR limits',
                        'Attribution analysis, benchmark relative',
                        'Investment committee oversight, external consultants'
                    ],
                    'Key Metrics': [
                        'Implementation shortfall, market impact',
                        'Tracking error, factor exposures, tail risk',
                        'Information ratio, alpha generation, fees',
                        'Policy compliance, fiduciary standards'
                    ]
                })
                
                st.dataframe(operational_framework, use_container_width=True)
                
                # Calculate institutional-level costs
                core_cost = equity_pct * 0.4 * 0.001  # 0.1% on core
                satellite_cost = equity_pct * 0.3 * 0.015  # 1.5% on satellites
                pe_cost = st.session_state.allocation['private_equity'] * 0.025  # 2.5% on PE
                total_equity_cost = (core_cost + satellite_cost + pe_cost) / 100
                
                st.metric("Blended Equity Management Fee", f"{total_equity_cost:.2%}")
    
    # === INSTITUTIONAL PORTFOLIO SUMMARY ===
    if 'allocation' in st.session_state:
        st.subheader("📊 Institutional Portfolio Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            final_allocation = pd.DataFrame({
                'Asset Class': ['Public Equity', 'Private Equity', 'Real Estate', 'Fixed Income', 'Alternatives/Cash'],
                'Allocation': [f"{st.session_state.allocation['public_equity']:.0f}%", 
                              f"{st.session_state.allocation['private_equity']:.0f}%",
                              f"{st.session_state.allocation['real_estate']:.0f}%",
                              f"{st.session_state.allocation['fixed_income']:.0f}%",
                              f"{st.session_state.allocation['alternatives']:.0f}%"],
                'Expected Return': ['7.5%', '9.2%', '6.8%', '4.2%', '5.5%'],
                'Liquidity': ['Daily', 'Quarterly', 'Monthly', 'Daily', 'Variable'],
                'Risk Level': ['High', 'Very High', 'Medium-High', 'Low', 'Medium']
            })
            
            st.dataframe(final_allocation, use_container_width=True)
        
        with summary_col2:
            institution = st.session_state.institution_profile
            risk = st.session_state.risk_assessment
            alloc = st.session_state.allocation
            
            st.markdown(f"""
            **Portfolio Characteristics:**
            - **Institution Type:** {institution['type']}
            - **Strategy Classification:** {risk['strategy']}
            - **Expected Return:** {alloc['expected_return']:.1%}
            - **Expected Volatility:** {alloc['volatility']:.1%}
            - **Sharpe Ratio (Est.):** {(alloc['expected_return'] - 0.035)/alloc['volatility']:.2f}
            - **Tracking Error Budget:** {risk['tracking_error']}
            """)
            
            st.markdown("""
            **Implementation Roadmap:**
            1. **Investment Policy Statement** - Formal documentation of objectives and constraints
            2. **Manager Selection** - Due diligence process for external managers
            3. **Risk Management System** - Real-time monitoring and reporting infrastructure  
            4. **Governance Structure** - Investment committee and decision-making framework
            5. **Performance Measurement** - Attribution analysis and benchmark reporting
            6. **Rebalancing Protocol** - Systematic approach to maintaining target allocations
            """)
            
            st.markdown(f"""
            **Key Performance Indicators:**
            - **Benchmark:** Custom strategic benchmark based on allocation
            - **Target Alpha:** 0.5-2.0% annual outperformance
            - **Risk Budget:** Max {risk['var_limit']} monthly VaR
            - **Rebalancing:** Quarterly review, ±3% deviation triggers
            - **Reporting:** Monthly performance, quarterly attribution analysis
            """)

    # === ADDITIONAL INSTITUTIONAL CONSIDERATIONS ===
    st.subheader("🏛️ Advanced Institutional Considerations")
    
    advanced_tabs = st.tabs(["The Theory: Factor Investing", "The Theory: Risk Management", "The Theory: Governance"])
    
    with advanced_tabs[0]:
        st.markdown("#### Factor-Based Portfolio Construction")
        
        st.markdown("""
        **Modern institutional equity portfolios** increasingly utilize factor-based approaches to enhance returns 
        and manage risk beyond traditional market cap-weighted indices.
        """)
        
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            factor_framework = pd.DataFrame({
                'Factor': ['Value', 'Quality', 'Low Volatility', 'Momentum', 'Size', 'Profitability'],
                'Risk Premium': ['3-4%', '2-3%', '2-3%', '5-8%', '1-2%', '2-4%'],
                'Implementation': ['P/B, P/E screens', 'ROE, debt ratios', 'Low beta, stable earnings', 'Price momentum', 'Small-cap tilt', 'ROA, margins'],
                'Institutional Use': ['Value tilt', 'Quality overlay', 'Risk reduction', 'Tactical allocation', 'Completion portfolios', 'Fundamental indexing']
            })
            
            st.dataframe(factor_framework, use_container_width=True)
            
        with factor_col2:
            st.markdown("""
            **Factor Integration Strategies:**
            
            **Multi-Factor Approach**: Combine factors to reduce single-factor risk while maintaining diversified risk premium exposure.
            
            **Factor Timing**: Institutional investors may dynamically adjust factor exposures based on market cycles and valuation metrics.
            
            **Implementation Methods**:
            - Internal quantitative strategies
            - External factor-based managers  
            - Smart beta ETFs and index funds
            - Custom factor indices
            """)
    
    with advanced_tabs[1]:
        st.markdown("#### Institutional Risk Management Framework")
        
        risk_framework = pd.DataFrame({
            'Risk Type': ['Market Risk', 'Credit Risk', 'Liquidity Risk', 'Operational Risk', 'Model Risk'],
            'Key Metrics': ['VaR, Beta, Correlation', 'Credit ratings, Spread duration', 'Bid-ask spreads, Trading volume', 'Key person, Process risk', 'Backtesting, Parameter sensitivity'],
            'Monitoring': ['Daily', 'Monthly', 'Weekly', 'Quarterly', 'Quarterly'],
            'Limits/Controls': ['VaR limits, Beta ranges', 'Credit quality minimums', 'Liquidity buckets', 'Due diligence, Audits', 'Model validation, Stress tests']
        })
        
        st.dataframe(risk_framework, use_container_width=True)
        
        st.markdown("""
        **Advanced Risk Techniques:**
        - **Scenario Analysis**: Stress testing under various market conditions
        - **Factor Risk Models**: Multi-factor risk attribution and forecasting
        - **Tail Risk Management**: Managing extreme downside scenarios
        - **Dynamic Hedging**: Options and derivatives for portfolio protection
        """)
    
    with advanced_tabs[2]:
        st.markdown("#### Governance and Fiduciary Framework")
        
        governance_col1, governance_col2 = st.columns(2)
        
        with governance_col1:
            governance_structure = pd.DataFrame({
                'Governance Level': ['Board of Trustees', 'Investment Committee', 'CIO/Investment Staff', 'External Managers', 'Service Providers'],
                'Key Responsibilities': [
                    'Fiduciary oversight, Policy approval',
                    'Strategy decisions, Manager selection', 
                    'Implementation, Risk monitoring',
                    'Portfolio management, Research',
                    'Custody, Administration, Consulting'
                ],
                'Meeting Frequency': ['Quarterly', 'Monthly', 'Daily/Weekly', 'As needed', 'Ongoing'],
                'Key Decisions': [
                    'Asset allocation, Spending policy',
                    'Manager hiring/firing, Tactical moves',
                    'Rebalancing, Trade execution', 
                    'Security selection, Risk management',
                    'Operations, Reporting, Compliance'
                ]
            })
            
            st.dataframe(governance_structure, use_container_width=True)
        
        with governance_col2:
            st.markdown("""
            **Fiduciary Best Practices:**
            
            **Investment Policy Statement**: Comprehensive document outlining objectives, constraints, and implementation approach.
            
            **Due Diligence Process**: Systematic evaluation of investment managers including quantitative and qualitative factors.
            
            **Performance Measurement**: Regular attribution analysis to understand sources of returns and risk.
            
            **Documentation**: Detailed records of investment decisions and rationale for fiduciary protection.
            
            **Conflicts of Interest**: Clear policies and procedures for managing potential conflicts in investment decisions.
            """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
        <p><strong>🎓 Institutional Portfolio Construction: Theory to Practice</strong></p>
        <p>This simulation demonstrates how institutional investors systematically build sophisticated equity portfolios, 
        balancing theoretical frameworks with practical implementation constraints and fiduciary responsibilities.</p>
    </div>
    """, unsafe_allow_html=True)
-Medium']
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
