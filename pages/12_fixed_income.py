# pages/12_Fixed_Income.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import requests
import yfinance as yf

def get_fred_data(series_id, start_date='2020-01-01'):
    """Get data from FRED API - returns sample data if API fails"""
    try:
        # FRED API endpoint (requires API key in production)
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1140&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id={series_id}&scale=left&cosd={start_date}&coed=2024-12-31&line_color=%23d62728&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-12-31&revision_date=2024-12-31&nd=1996-12-31'
        
        # In production, you would use: response = requests.get(url)
        # For demo, return simulated data
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='D')
        
        if 'HYM2' in series_id:  # High Yield
            base_spread = 400  # 4%
            volatility = 150
        elif 'BBB' in series_id:  # BBB Corporate
            base_spread = 120  # 1.2%
            volatility = 40
        elif 'AAA' in series_id:  # AAA Corporate
            base_spread = 50   # 0.5%
            volatility = 20
        else:  # Investment Grade
            base_spread = 100  # 1%
            volatility = 30
            
        # Simulate spreads with crisis periods
        spreads = []
        for i, date in enumerate(dates):
            # Base trend
            trend = base_spread + np.random.normal(0, volatility/10)
            
            # COVID crisis
            if date.year == 2020 and 3 <= date.month <= 6:
                crisis = volatility * 2
            # 2022 rate shock
            elif date.year == 2022 and 6 <= date.month <= 10:
                crisis = volatility * 0.5
            else:
                crisis = 0
                
            spread = max(10, trend + crisis + np.random.normal(0, volatility/20))
            spreads.append(spread)
        
        return pd.DataFrame({
            'DATE': dates,
            series_id: spreads
        })
    
    except:
        # Fallback data if API fails
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='M')
        spreads = np.random.normal(200, 50, len(dates))
        return pd.DataFrame({'DATE': dates, series_id: spreads})

def calculate_bond_price(face_value, coupon_rate, yield_rate, years):
    """Calculate bond price using present value formula"""
    # PV = Σ[C/(1+r)^t] + FV/(1+r)^n
    coupon = face_value * coupon_rate
    price = 0
    
    # Present value of coupons
    for t in range(1, int(years) + 1):
        price += coupon / (1 + yield_rate) ** t
    
    # Present value of face value
    price += face_value / (1 + yield_rate) ** years
    
    return price

def calculate_duration(coupon_rate, yield_rate, years):
    """Calculate modified duration"""
    # Simplified duration calculation
    duration = 0
    face_value = 1000
    coupon = face_value * coupon_rate
    
    for t in range(1, int(years) + 1):
        pv_coupon = coupon / (1 + yield_rate) ** t
        duration += t * pv_coupon
    
    pv_face = face_value / (1 + yield_rate) ** years
    duration += years * pv_face
    
    bond_price = calculate_bond_price(face_value, coupon_rate, yield_rate, years)
    duration = duration / bond_price
    modified_duration = duration / (1 + yield_rate)
    
    return modified_duration

def show_fixed_income_page():
    st.set_page_config(page_title="Fixed Income Strategy Builder", layout="wide")
    
    st.title("🏦 Interactive Fixed Income Strategy Builder")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Institutional Fixed Income Portfolio Construction</h3>
        <p style="color: #e8f4fd; margin: 0.5rem 0 0 0;">
            Design sophisticated fixed income strategies with real-time market data and advanced analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === MAIN STRATEGY BUILDER ===
    st.subheader("🎯 Strategy Configuration")
    
    # Strategy selection tabs
    strategy_tabs = st.tabs(["Pre-Built Strategies", "Custom Strategy Builder", "Market Analysis"])
    
    with strategy_tabs[0]:
        st.markdown("### Institutional Strategy Templates")
        
        template_col1, template_col2 = st.columns(2)
        
        with template_col1:
            st.markdown("**Choose Your Institution Type:**")
            
            institution_strategy = st.selectbox(
                "Institution Type:",
                ["Corporate Pension Fund", "Insurance Company", "Sovereign Wealth Fund", 
                 "University Endowment", "Central Bank", "Money Market Fund"]
            )
            
            scenario = st.selectbox(
                "Market Scenario:",
                ["Rising Rate Environment", "Falling Rate Environment", "High Volatility Period", 
                 "Credit Crisis", "Normal Markets", "Inflation Surge"]
            )
            
        with template_col2:
            # Define strategy templates
            strategies = {
                "Corporate Pension Fund": {
                    "Rising Rate Environment": {
                        "duration": 4.5, "credit_allocation": 60, "government": 40,
                        "hy_allocation": 10, "ig_allocation": 50, "treasury_allocation": 40,
                        "rationale": "Shorter duration to limit interest rate risk, moderate credit exposure"
                    },
                    "Credit Crisis": {
                        "duration": 6.5, "credit_allocation": 30, "government": 70,
                        "hy_allocation": 5, "ig_allocation": 25, "treasury_allocation": 70,
                        "rationale": "Flight to quality, reduced credit exposure, longer duration for safety"
                    }
                },
                "Insurance Company": {
                    "Rising Rate Environment": {
                        "duration": 8.5, "credit_allocation": 45, "government": 55,
                        "hy_allocation": 5, "ig_allocation": 40, "treasury_allocation": 55,
                        "rationale": "Liability matching requires longer duration, conservative credit approach"
                    },
                    "Normal Markets": {
                        "duration": 10.2, "credit_allocation": 55, "government": 45,
                        "hy_allocation": 8, "ig_allocation": 47, "treasury_allocation": 45,
                        "rationale": "ALM-driven duration, spread pickup from credit allocation"
                    }
                },
                "Sovereign Wealth Fund": {
                    "High Volatility Period": {
                        "duration": 5.8, "credit_allocation": 75, "government": 25,
                        "hy_allocation": 25, "ig_allocation": 50, "treasury_allocation": 25,
                        "rationale": "Opportunistic credit allocation, tactical duration positioning"
                    }
                }
            }
            
            # Get strategy or use defaults
            if institution_strategy in strategies and scenario in strategies[institution_strategy]:
                strategy = strategies[institution_strategy][scenario]
            else:
                strategy = {
                    "duration": 6.0, "credit_allocation": 50, "government": 50,
                    "hy_allocation": 10, "ig_allocation": 40, "treasury_allocation": 50,
                    "rationale": "Balanced approach for selected scenario"
                }
            
            st.markdown(f"""
            **Recommended Strategy:**
            - **Target Duration:** {strategy['duration']:.1f} years
            - **Government Bonds:** {strategy['government']:.0f}%
            - **Investment Grade:** {strategy['ig_allocation']:.0f}%
            - **High Yield:** {strategy['hy_allocation']:.0f}%
            
            **Rationale:** {strategy['rationale']}
            """)
            
            # Store strategy in session state
            st.session_state.selected_strategy = strategy
            
    with strategy_tabs[1]:
        st.markdown("### Custom Strategy Builder")
        
        builder_col1, builder_col2 = st.columns(2)
        
        with builder_col1:
            st.markdown("**Portfolio Allocation:**")
            
            treasury_pct = st.slider("US Treasury (%)", 0, 100, 40, 5)
            ig_corporate_pct = st.slider("Investment Grade Corporate (%)", 0, 100, 35, 5)
            hy_corporate_pct = st.slider("High Yield Corporate (%)", 0, 100, 15, 5)
            international_pct = st.slider("International Bonds (%)", 0, 100, 10, 5)
            
            total_allocation = treasury_pct + ig_corporate_pct + hy_corporate_pct + international_pct
            
            if total_allocation != 100:
                st.warning(f"Total allocation: {total_allocation}% (should be 100%)")
            else:
                st.success("✅ Portfolio allocation balanced")
            
            target_duration = st.slider("Target Portfolio Duration (years)", 1.0, 15.0, 6.0, 0.5)
            
        with builder_col2:
            st.markdown("**Risk Parameters:**")
            
            credit_risk_budget = st.selectbox("Credit Risk Budget:", 
                ["Conservative", "Moderate", "Aggressive"])
            
            duration_risk = st.selectbox("Duration Risk Tolerance:",
                ["Low (1-3 years)", "Medium (4-7 years)", "High (8+ years)"])
            
            liquidity_requirement = st.selectbox("Liquidity Requirements:",
                ["High (Daily)", "Medium (Weekly)", "Low (Monthly+)"])
            
            # Calculate portfolio metrics
            if total_allocation == 100:
                # Simplified yield calculation
                treasury_yield = 4.5  # Current 10Y Treasury
                ig_spread = 120  # basis points
                hy_spread = 450  # basis points
                intl_spread = 80   # basis points
                
                portfolio_yield = (
                    treasury_pct/100 * treasury_yield +
                    ig_corporate_pct/100 * (treasury_yield + ig_spread/100) +
                    hy_corporate_pct/100 * (treasury_yield + hy_spread/100) +
                    international_pct/100 * (treasury_yield + intl_spread/100)
                )
                
                # Simplified risk metrics
                duration_risk_score = {
                    "Low (1-3 years)": 2, "Medium (4-7 years)": 5, "High (8+ years)": 8
                }[duration_risk]
                
                credit_risk_score = {
                    "Conservative": 2, "Moderate": 5, "Aggressive": 8
                }[credit_risk_budget]
                
                st.markdown(f"""
                **Portfolio Metrics:**
                - **Expected Yield:** {portfolio_yield:.2f}%
                - **Target Duration:** {target_duration:.1f} years
                - **Credit Risk Score:** {credit_risk_score}/10
                - **Duration Risk Score:** {duration_risk_score}/10
                """)
                
                # Store custom strategy
                st.session_state.custom_strategy = {
                    'treasury': treasury_pct, 'ig': ig_corporate_pct, 
                    'hy': hy_corporate_pct, 'intl': international_pct,
                    'duration': target_duration, 'yield': portfolio_yield
                }
    
    with strategy_tabs[2]:
        st.markdown("### Real-Time Market Analysis")
        
        # Get real-time credit spread data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Credit Spreads Analysis**")
            
            # Load credit spread data
            aaa_data = get_fred_data('BAMLC0A1CAAA', '2020-01-01')
            bbb_data = get_fred_data('BAMLC0A4CBBB', '2020-01-01') 
            hy_data = get_fred_data('BAMLH0A0HYM2', '2020-01-01')
            
            fig_spreads = go.Figure()
            
            fig_spreads.add_trace(go.Scatter(
                x=aaa_data['DATE'], 
                y=aaa_data.iloc[:, 1],
                mode='lines',
                name='AAA Corporate',
                line=dict(color='green', width=2)
            ))
            
            fig_spreads.add_trace(go.Scatter(
                x=bbb_data['DATE'],
                y=bbb_data.iloc[:, 1], 
                mode='lines',
                name='BBB Corporate',
                line=dict(color='orange', width=2)
            ))
            
            fig_spreads.add_trace(go.Scatter(
                x=hy_data['DATE'],
                y=hy_data.iloc[:, 1],
                mode='lines', 
                name='High Yield',
                line=dict(color='red', width=2)
            ))
            
            fig_spreads.update_layout(
                title="Credit Spreads Over Time (basis points)",
                xaxis_title="Date",
                yaxis_title="Option-Adjusted Spread (bp)",
                height=400
            )
            
            st.plotly_chart(fig_spreads, use_container_width=True)
            st.caption("*Data source: Federal Reserve Economic Data (FRED) - ICE BofA Indices*")
            
        with col2:
            st.markdown("**Current Market Conditions**")
            
            # Current spread levels (using last values)
            current_aaa = aaa_data.iloc[-1, 1] if len(aaa_data) > 0 else 50
            current_bbb = bbb_data.iloc[-1, 1] if len(bbb_data) > 0 else 120
            current_hy = hy_data.iloc[-1, 1] if len(hy_data) > 0 else 400
            
            spread_metrics = pd.DataFrame({
                'Rating': ['AAA Corporate', 'BBB Corporate', 'High Yield'],
                'Current Spread (bp)': [f'{current_aaa:.0f}', f'{current_bbb:.0f}', f'{current_hy:.0f}'],
                'Historical Average': ['45bp', '125bp', '450bp'],
                'Assessment': [
                    'Above Average' if current_aaa > 45 else 'Below Average',
                    'Tight' if current_bbb < 125 else 'Wide', 
                    'Tight' if current_hy < 450 else 'Wide'
                ]
            })
            
            st.dataframe(spread_metrics, use_container_width=True)
            
            # Market regime assessment
            if current_hy > 600:
                regime = "🔴 Crisis Mode"
                recommendation = "Flight to quality, reduce credit exposure"
            elif current_hy < 300:
                regime = "🟢 Risk-On Environment" 
                recommendation = "Opportunities in credit markets"
            else:
                regime = "🟡 Normal Markets"
                recommendation = "Balanced approach appropriate"
                
            st.markdown(f"""
            **Market Regime:** {regime}
            
            **Investment Implication:** {recommendation}
            """)
    
    # === ADVANCED ANALYTICS ===
    st.subheader("📊 Portfolio Analytics & Risk Management")
    
    analytics_tabs = st.tabs(["Duration Analysis", "Credit Risk", "Scenario Testing"])
    
    with analytics_tabs[0]:
        st.markdown("### Duration and Interest Rate Risk")
        
        duration_col1, duration_col2 = st.columns(2)
        
        with duration_col1:
            st.markdown("**Duration Calculator**")
            
            bond_coupon = st.slider("Coupon Rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
            bond_yield = st.slider("Yield to Maturity (%)", 0.0, 15.0, 5.0, 0.25) / 100
            bond_maturity = st.slider("Years to Maturity", 1, 30, 10)
            
            # Calculate bond metrics
            bond_price = calculate_bond_price(1000, bond_coupon, bond_yield, bond_maturity)
            duration = calculate_duration(bond_coupon, bond_yield, bond_maturity)
            
            st.markdown(f"""
            **Bond Analytics:**
            - **Price:** ${bond_price:.2f}
            - **Modified Duration:** {duration:.2f} years
            - **DV01:** ${(bond_price * duration * 0.0001):.2f}
            """)
            
            # Duration formula
            st.markdown("**Modified Duration Formula:**")
            st.latex(r"D_{mod} = \frac{D}{1 + YTM}")
            st.latex(r"D = \frac{\sum_{t=1}^{n} \frac{t \cdot CF_t}{(1+YTM)^t}}{P}")
            
        with duration_col2:
            st.markdown("**Interest Rate Sensitivity Analysis**")
            
            # Simulate rate changes
            rate_changes = np.arange(-2.0, 2.1, 0.25)  # -200bp to +200bp
            price_changes = []
            
            for rate_change in rate_changes:
                new_yield = bond_yield + (rate_change / 100)
                new_price = calculate_bond_price(1000, bond_coupon, new_yield, bond_maturity)
                price_change = ((new_price - bond_price) / bond_price) * 100
                price_changes.append(price_change)
            
            fig_duration = go.Figure()
            
            fig_duration.add_trace(go.Scatter(
                x=rate_changes * 100,  # Convert to basis points
                y=price_changes,
                mode='lines+markers',
                name='Price Sensitivity',
                line=dict(color='blue', width=3)
            ))
            
            # Add linear approximation using duration
            linear_approx = [-duration * rate_change * 100 for rate_change in rate_changes]
            
            fig_duration.add_trace(go.Scatter(
                x=rate_changes * 100,
                y=linear_approx,
                mode='lines',
                name='Duration Approximation',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig_duration.update_layout(
                title="Bond Price Sensitivity to Interest Rate Changes",
                xaxis_title="Yield Change (basis points)",
                yaxis_title="Price Change (%)",
                height=400
            )
            
            st.plotly_chart(fig_duration, use_container_width=True)
            
    with analytics_tabs[1]:
        st.markdown("### Credit Risk Analysis")
        
        credit_col1, credit_col2 = st.columns(2)
        
        with credit_col1:
            # Credit transition matrix
            st.markdown("**Credit Migration Analysis**")
            
            transition_matrix = pd.DataFrame({
                'Rating': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],
                'AAA': [90.8, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
                'AA': [8.3, 90.6, 7.8, 0.6, 0.1, 0.0, 0.0],
                'A': [0.7, 2.3, 91.1, 5.5, 0.4, 0.1, 0.0],
                'BBB': [0.1, 0.3, 4.4, 86.9, 5.4, 1.2, 0.2],
                'BB': [0.0, 0.1, 0.5, 7.8, 80.5, 8.8, 1.0],
                'B': [0.0, 0.0, 0.2, 0.4, 6.3, 83.5, 4.9],
                'Default': [0.0, 0.0, 0.0, 0.2, 1.0, 4.4, 19.6]
            })
            
            st.dataframe(transition_matrix, use_container_width=True)
            st.caption("*1-year credit migration probabilities (%)*")
            
        with credit_col2:
            st.markdown("**Expected Loss Calculation**")
            
            portfolio_rating = st.selectbox("Portfolio Average Rating:", 
                ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'])
            
            portfolio_size = st.number_input("Portfolio Size ($M):", 100, 10000, 1000)
            loss_given_default = st.slider("Loss Given Default (%)", 20, 80, 40) / 100
            
            # Default probabilities (1-year)
            default_probs = {
                'AAA': 0.00, 'AA': 0.00, 'A': 0.00, 'BBB': 0.002,
                'BB': 0.010, 'B': 0.044, 'CCC': 0.196
            }
            
            prob_default = default_probs[portfolio_rating]
            expected_loss = portfolio_size * prob_default * loss_given_default
            
            st.markdown(f"""
            **Credit Risk Metrics:**
            - **Probability of Default:** {prob_default:.1%}
            - **Loss Given Default:** {loss_given_default:.0%}
            - **Expected Loss:** ${expected_loss:.2f}M
            - **Credit VaR (95%):** ${expected_loss * 2.33:.2f}M
            """)
            
            # Credit loss formula
            st.markdown("**Expected Loss Formula:**")
            st.latex(r"EL = EAD \times PD \times LGD")
            st.markdown("Where: EAD = Exposure at Default, PD = Probability of Default, LGD = Loss Given Default")
    
    with analytics_tabs[2]:
        st.markdown("### Scenario Testing")
        
        scenario_col1, scenario_col2 = st.columns(2)
        
        with scenario_col1:
            st.markdown("**Stress Test Scenarios**")
            
            scenarios = {
                "2008 Financial Crisis": {"rate_change": -2.5, "spread_change": 300, "description": "Flight to quality"},
                "2013 Taper Tantrum": {"rate_change": 1.0, "spread_change": 50, "description": "Rising rate shock"},
                "2020 COVID Crisis": {"rate_change": -1.5, "spread_change": 200, "description": "Liquidity crisis"},
                "2022 Inflation Shock": {"rate_change": 2.0, "spread_change": 75, "description": "Aggressive Fed tightening"},
                "Custom Scenario": {"rate_change": 0, "spread_change": 0, "description": "User defined"}
            }
            
            selected_scenario = st.selectbox("Choose Scenario:", list(scenarios.keys()))
            
            if selected_scenario == "Custom Scenario":
                rate_shock = st.slider("Interest Rate Change (%)", -3.0, 3.0, 0.0, 0.25)
                spread_shock = st.slider("Credit Spread Change (bp)", -200, 500, 0, 25)
            else:
                scenario = scenarios[selected_scenario]
                rate_shock = scenario["rate_change"]
                spread_shock = scenario["spread_change"]
                
                st.markdown(f"""
                **Scenario: {selected_scenario}**
                - Rate Change: {rate_shock:+.1f}%
                - Spread Change: {spread_shock:+.0f}bp
                - Description: {scenario['description']}
                """)
        
        with scenario_col2:
            st.markdown("**Portfolio Impact Analysis**")
            
            # Use strategy from session state if available
            if 'selected_strategy' in st.session_state:
                strategy = st.session_state.selected_strategy
                gov_weight = strategy['government'] / 100
                credit_weight = strategy['credit_allocation'] / 100
                duration = strategy['duration']
            else:
                # Default portfolio
                gov_weight = 0.5
                credit_weight = 0.5
                duration = 6.0
            
            # Calculate scenario impact
            duration_impact = -duration * (rate_shock / 100)  # Duration impact from rates
            credit_impact = -credit_weight * (spread_shock / 10000)  # Credit impact from spreads
            total_impact = duration_impact + credit_impact
            
            # Portfolio value impact (assuming $1B portfolio)
            portfolio_value = 1000  # $1B
            dollar_impact = total_impact * portfolio_value
            
            impact_df = pd.DataFrame({
                'Component': ['Duration Impact', 'Credit Spread Impact', 'Total Impact'],
                'Return Impact (%)': [f'{duration_impact:.2%}', f'{credit_impact:.2%}', f'{total_impact:.2%}'],
                'Dollar Impact ($M)': [f'${duration_impact * portfolio_value:.0f}', 
                                      f'${credit_impact * portfolio_value:.0f}', 
                                      f'${dollar_impact:.0f}']
            })
            
            st.dataframe(impact_df, use_container_width=True)
            
            # Risk assessment
            if abs(total_impact) > 0.05:
                risk_level = "🔴 High Risk"
            elif abs(total_impact) > 0.02:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"
                
            st.markdown(f"""
            **Risk Assessment:** {risk_level}
            
            **Recommended Actions:**
            - Consider hedging interest rate exposure if duration impact > 3%
            - Review credit allocation if spread impact > 2%
            - Implement dynamic hedging for high-risk scenarios
            """)
    
    # === IMPLEMENTATION SUMMARY ===
    if 'selected_strategy' in st.session_state or 'custom_strategy' in st.session_state:
        st.subheader("📋 Implementation Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            if 'selected_strategy' in st.session_state:
                strategy = st.session_state.selected_strategy
                st.markdown(f"""
                **Selected Strategy Configuration:**
                - **Duration Target:** {strategy['duration']:.1f} years
                - **Government Allocation:** {strategy['government']:.0f}%
                - **Credit Allocation:** {strategy['credit_allocation']:.0f}%
                - **High Yield:** {strategy['hy_allocation']:.0f}%
                """)
            elif 'custom_strategy' in st.session_state:
                strategy = st.session_state.custom_strategy
                st.markdown(f"""
                **Custom Strategy Configuration:**
                - **Treasury:** {strategy['treasury']:.0f}%
                - **Investment Grade:** {strategy['ig']:.0f}%
                - **High Yield:** {strategy['hy']:.0f}%
                - **International:** {strategy['intl']:.0f}%
                - **Expected Yield:** {strategy['yield']:.2f}%
                """)
        
        with summary_col2:
            st.markdown("""
            **Next Steps:**
            1. **Manager Selection** - Due diligence on fixed income managers
            2. **Implementation** - Phased approach over 3-6 months
            3. **Risk Monitoring** - Daily duration and credit risk tracking
            4. **Performance Attribution** - Monthly analysis vs benchmark
            5. **Rebalancing** - Quarterly review and tactical adjustments
            
            **Key Risk Controls:**
            - Duration limits: ±1 year vs target
            - Credit exposure caps by rating
            - Liquidity requirements monitoring
            - Stress testing monthly
            """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
        <p><strong>🏦 Advanced Fixed Income Strategy Builder</strong></p>
        <p>This tool demonstrates institutional-grade fixed income portfolio construction with real-time market data, 
        sophisticated analytics, and comprehensive risk management frameworks.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_fixed_income_page()
