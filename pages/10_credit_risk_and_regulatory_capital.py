# pages/11_Credit_risk_management.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import norm, chi2
from scipy.optimize import minimize_scalar
import yfinance as yf
from datetime import datetime, timedelta

def calculate_merton_model(S, D, T, r, sigma_equity):
    """
    Calculate default probability using Merton structural model
    
    Parameters:
    S: Current equity value (market cap)
    D: Debt value (face value)
    T: Time to maturity
    r: Risk-free rate
    sigma_equity: Equity volatility
    
    Returns:
    Default probability, asset value, asset volatility
    """
    
    # Initial guess for asset value and volatility
    V0_guess = S + D
    sigma_V_guess = sigma_equity * S / V0_guess
    
    def merton_equations(params):
        V0, sigma_V = params
        
        # Calculate d1 and d2
        d1 = (np.log(V0/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        
        # Merton equity value
        equity_value = V0 * norm.cdf(d1) - D * np.exp(-r*T) * norm.cdf(d2)
        
        # Merton equity volatility
        equity_vol = (V0 * norm.pdf(d1) * sigma_V) / equity_value if equity_value > 0 else 0
        
        # Error terms
        equity_error = (equity_value - S)**2
        vol_error = (equity_vol - sigma_equity)**2
        
        return equity_error + vol_error
    
    # Solve for asset value and volatility
    from scipy.optimize import minimize
    result = minimize(merton_equations, [V0_guess, sigma_V_guess], 
                     bounds=[(S*0.5, S*5), (0.01, 1.0)], method='L-BFGS-B')
    
    V0_optimal, sigma_V_optimal = result.x
    
    # Calculate default probability
    d2 = (np.log(V0_optimal/D) + (r - 0.5*sigma_V_optimal**2)*T) / (sigma_V_optimal * np.sqrt(T))
    default_prob = norm.cdf(-d2)
    
    return default_prob, V0_optimal, sigma_V_optimal

def simulate_credit_portfolio_losses(n_obligors, default_probs, correlations, recovery_rates, exposures, n_simulations=10000):
    """
    Monte Carlo simulation of credit portfolio losses using factor model
    """
    
    # Convert to numpy arrays
    default_probs = np.array(default_probs)
    recovery_rates = np.array(recovery_rates)
    exposures = np.array(exposures)
    
    # Factor loadings (simplified: single factor model)
    factor_loadings = np.sqrt(correlations)
    idiosyncratic_weights = np.sqrt(1 - correlations)
    
    portfolio_losses = []
    
    for sim in range(n_simulations):
        # Generate common factor
        common_factor = np.random.standard_normal()
        
        # Generate idiosyncratic factors
        idiosyncratic_factors = np.random.standard_normal(n_obligors)
        
        # Asset returns for each obligor
        asset_returns = factor_loadings * common_factor + idiosyncratic_weights * idiosyncratic_factors
        
        # Default thresholds
        default_thresholds = norm.ppf(default_probs)
        
        # Determine defaults
        defaults = asset_returns < default_thresholds
        
        # Calculate losses
        losses = defaults * (1 - recovery_rates) * exposures
        portfolio_loss = np.sum(losses)
        
        portfolio_losses.append(portfolio_loss)
    
    return np.array(portfolio_losses)

def calculate_regulatory_capital_ratios(total_assets, tier1_capital, tier2_capital, rwa):
    """
    Calculate key regulatory capital ratios
    """
    total_capital = tier1_capital + tier2_capital
    
    ratios = {
        'CET1_Ratio': (tier1_capital / rwa) * 100,  # Common Equity Tier 1
        'Tier1_Ratio': (tier1_capital / rwa) * 100,  # Tier 1 Capital Ratio
        'Total_Capital_Ratio': (total_capital / rwa) * 100,  # Total Capital Ratio
        'Leverage_Ratio': (tier1_capital / total_assets) * 100  # Leverage Ratio
    }
    
    return ratios

def create_rating_transition_matrix():
    """
    Create a sample 1-year rating transition matrix
    """
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']
    
    # Sample transition matrix (percentages)
    transition_matrix = np.array([
        [90.45, 8.50, 0.70, 0.25, 0.07, 0.02, 0.01, 0.00],  # AAA
        [0.84, 91.20, 6.84, 0.75, 0.22, 0.10, 0.03, 0.02],  # AA
        [0.07, 2.30, 91.40, 5.20, 0.70, 0.23, 0.07, 0.03],  # A
        [0.02, 0.35, 5.50, 88.60, 4.20, 0.98, 0.25, 0.10],  # BBB
        [0.01, 0.08, 0.55, 7.20, 83.30, 7.40, 1.10, 0.36],  # BB
        [0.00, 0.03, 0.15, 0.68, 7.90, 83.70, 6.50, 1.04],  # B
        [0.00, 0.01, 0.05, 0.22, 0.98, 7.60, 82.64, 8.50],  # CCC
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 100.00]  # Default
    ])
    
    return pd.DataFrame(transition_matrix, index=ratings, columns=ratings)

def calculate_unexpected_loss_contributions(exposures, default_probs, correlations, recovery_rates):
    """
    Calculate unexpected loss contributions for portfolio risk allocation
    """
    n = len(exposures)
    exposures = np.array(exposures)
    default_probs = np.array(default_probs)
    recovery_rates = np.array(recovery_rates)
    lgd = 1 - recovery_rates
    
    # Individual unexpected losses
    individual_ul = exposures * lgd * np.sqrt(default_probs * (1 - default_probs))
    
    # Portfolio unexpected loss (simplified)
    correlation_matrix = np.full((n, n), correlations) + np.eye(n) * (1 - correlations)
    portfolio_ul = np.sqrt(np.dot(individual_ul, np.dot(correlation_matrix, individual_ul)))
    
    # Marginal contributions
    marginal_contributions = (correlation_matrix @ individual_ul) / portfolio_ul
    
    # Component contributions  
    component_contributions = individual_ul * marginal_contributions
    
    return {
        'individual_ul': individual_ul,
        'portfolio_ul': portfolio_ul,
        'marginal_contributions': marginal_contributions,
        'component_contributions': component_contributions
    }

def show_credit_risk_management_page():
    st.set_page_config(page_title="Credit Risk Management & Regulatory Capital", layout="wide")
    
    st.title("🏦 Credit Risk Management & Regulatory Capital Framework")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Comprehensive Credit Risk Assessment & Capital Management</h3>
        <p style="color: #e8f4fd; margin: 0.5rem 0 0 0;">
            Master credit risk modeling from individual exposures to portfolio-level capital requirements under Basel III/IV frameworks
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === SECTION 1: FUNDAMENTALS OF CREDIT RISK ===
    st.subheader("📊 Fundamentals of Credit Risk")
    
    with st.expander("🎯 Core Credit Risk Components", expanded=True):
        st.markdown("""
        ### Understanding Credit Risk Dimensions
        
        Credit risk is multifaceted, encompassing several interconnected components that financial institutions must measure and manage:
        
        **1. Default Risk**
        - The fundamental risk that a borrower fails to meet contractual obligations
        - Binary event with asymmetric payoffs (limited upside, significant downside)
        - Probability of Default (PD) is the key metric, typically expressed as an annual percentage
        
        **2. Credit Spread Risk** 
        - Market-driven changes in credit spreads due to perceived creditworthiness shifts
        - Affects mark-to-market valuations even without actual defaults
        - Particularly relevant for tradable instruments and fair value accounting
        
        **3. Migration Risk**
        - Risk of credit rating downgrades affecting instrument valuations
        - Can trigger regulatory breaches or forced selling for certain investors
        - Especially critical around investment grade/high yield boundaries
        
        **4. Recovery Risk**
        - Uncertainty in Loss Given Default (LGD) upon actual default events
        - Recovery rates vary by seniority, collateral, jurisdiction, and market conditions
        - Often correlated with default rates (lower recoveries during crisis periods)
        
        **5. Concentration Risk**
        - Risk from inadequate diversification across obligors, sectors, or geographies
        - Can amplify losses through correlation effects during stressed conditions
        - Managed through exposure limits and portfolio diversification requirements
        """)
    
    # === SECTION 2: STRUCTURAL MODELS - MERTON FRAMEWORK ===
    st.subheader("🏗️ Structural Models: The Merton Framework")
    
    st.markdown("""
    **The Merton Model** treats corporate equity as a call option on the firm's assets, providing an economic foundation 
    for linking default risk to capital structure and market observables. This approach views default as occurring when 
    the firm's asset value falls below its debt obligations at maturity.
    """)
    
    # Merton Model Calculator
    with st.container():
        st.markdown("#### Merton Model Implementation")
        
        merton_col1, merton_col2 = st.columns(2)
        
        with merton_col1:
            st.markdown("**Company Parameters**")
            
            # Sample companies for demonstration
            sample_companies = {
                "Custom": {"market_cap": 50.0, "debt": 30.0, "volatility": 0.25},
                "Large Cap Stable": {"market_cap": 200.0, "debt": 80.0, "volatility": 0.20},
                "Mid Cap Growth": {"market_cap": 15.0, "debt": 10.0, "volatility": 0.35},
                "High Leverage": {"market_cap": 25.0, "debt": 40.0, "volatility": 0.45}
            }
            
            company_type = st.selectbox("Company Profile", list(sample_companies.keys()))
            
            if company_type == "Custom":
                market_cap = st.number_input("Market Capitalization ($B)", value=50.0, min_value=1.0, step=1.0)
                debt_value = st.number_input("Debt Value ($B)", value=30.0, min_value=1.0, step=1.0)
                equity_vol = st.slider("Equity Volatility", 0.1, 0.8, 0.25, 0.05)
            else:
                params = sample_companies[company_type]
                market_cap = params["market_cap"]
                debt_value = params["debt"]
                equity_vol = params["volatility"]
                
                st.info(f"**{company_type}:** Market Cap ${market_cap}B, Debt ${debt_value}B, Vol {equity_vol:.1%}")
            
            time_horizon = st.slider("Time Horizon (years)", 0.25, 5.0, 1.0, 0.25)
            risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 8.0, 3.0, 0.25) / 100
        
        with merton_col2:
            st.markdown("**Model Results**")
            
            # Calculate Merton model results
            try:
                default_prob, asset_value, asset_vol = calculate_merton_model(
                    market_cap, debt_value, time_horizon, risk_free_rate, equity_vol
                )
                
                # Display results
                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Default Probability", f"{default_prob:.2%}")
                result_col2.metric("Distance to Default", f"{norm.ppf(1-default_prob):.2f}")
                result_col1.metric("Implied Asset Value", f"${asset_value:.1f}B")
                result_col2.metric("Asset Volatility", f"{asset_vol:.1%}")
                
                # Leverage ratio
                leverage = debt_value / asset_value
                st.metric("Asset Leverage", f"{leverage:.1%}")
                
                # Interpretation
                if default_prob < 0.01:
                    risk_level = "🟢 Very Low Risk"
                elif default_prob < 0.05:
                    risk_level = "🟡 Low Risk"
                elif default_prob < 0.15:
                    risk_level = "🟠 Moderate Risk" 
                else:
                    risk_level = "🔴 High Risk"
                    
                st.info(f"**Risk Assessment:** {risk_level}")
                
            except Exception as e:
                st.error(f"Model calculation error: {str(e)}")
    
    # Merton Model Sensitivity Analysis
    st.markdown("#### Sensitivity Analysis: Default Probability vs Key Parameters")
    
    # Create sensitivity charts
    fig_merton = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Asset Volatility Impact', 'Leverage Impact', 'Time Horizon Impact', 'Interest Rate Impact'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Base case parameters
    base_S, base_D, base_T, base_r, base_vol = market_cap, debt_value, time_horizon, risk_free_rate, equity_vol
    
    # Volatility sensitivity
    vol_range = np.linspace(0.1, 0.6, 20)
    vol_defaults = []
    for vol in vol_range:
        try:
            dp, _, _ = calculate_merton_model(base_S, base_D, base_T, base_r, vol)
            vol_defaults.append(dp)
        except:
            vol_defaults.append(np.nan)
    
    fig_merton.add_trace(
        go.Scatter(x=vol_range*100, y=np.array(vol_defaults)*100, mode='lines', name='Vol Sensitivity'),
        row=1, col=1
    )
    
    # Leverage sensitivity (varying debt)
    debt_range = np.linspace(base_D*0.5, base_D*2.0, 20)
    debt_defaults = []
    for debt in debt_range:
        try:
            dp, _, _ = calculate_merton_model(base_S, debt, base_T, base_r, base_vol)
            debt_defaults.append(dp)
        except:
            debt_defaults.append(np.nan)
    
    fig_merton.add_trace(
        go.Scatter(x=debt_range/base_S*100, y=np.array(debt_defaults)*100, mode='lines', name='Leverage Sensitivity'),
        row=1, col=2
    )
    
    # Time sensitivity
    time_range = np.linspace(0.25, 5.0, 20)
    time_defaults = []
    for t in time_range:
        try:
            dp, _, _ = calculate_merton_model(base_S, base_D, t, base_r, base_vol)
            time_defaults.append(dp)
        except:
            time_defaults.append(np.nan)
    
    fig_merton.add_trace(
        go.Scatter(x=time_range, y=np.array(time_defaults)*100, mode='lines', name='Time Sensitivity'),
        row=2, col=1
    )
    
    # Interest rate sensitivity
    rate_range = np.linspace(0.01, 0.08, 20)
    rate_defaults = []
    for rate in rate_range:
        try:
            dp, _, _ = calculate_merton_model(base_S, base_D, base_T, rate, base_vol)
            rate_defaults.append(dp)
        except:
            rate_defaults.append(np.nan)
    
    fig_merton.add_trace(
        go.Scatter(x=rate_range*100, y=np.array(rate_defaults)*100, mode='lines', name='Rate Sensitivity'),
        row=2, col=2
    )
    
    # Update layout
    fig_merton.update_xaxes(title_text="Volatility (%)", row=1, col=1)
    fig_merton.update_xaxes(title_text="Debt/Equity (%)", row=1, col=2)  
    fig_merton.update_xaxes(title_text="Time (years)", row=2, col=1)
    fig_merton.update_xaxes(title_text="Risk-free Rate (%)", row=2, col=2)
    
    fig_merton.update_yaxes(title_text="Default Prob (%)", row=1, col=1)
    fig_merton.update_yaxes(title_text="Default Prob (%)", row=1, col=2)
    fig_merton.update_yaxes(title_text="Default Prob (%)", row=2, col=1) 
    fig_merton.update_yaxes(title_text="Default Prob (%)", row=2, col=2)
    
    fig_merton.update_layout(height=600, showlegend=False, title_text="Merton Model Sensitivity Analysis")
    st.plotly_chart(fig_merton, use_container_width=True)
    
    with st.expander("📖 Understanding the Merton Model"):
        st.markdown("""
        ### Model Mechanics and Interpretation
        
        The **Merton structural model** provides economic intuition by linking default directly to firm fundamentals:
        
        **Core Equation:**
        
        The firm defaults when Asset Value < Debt Value at maturity. The default probability is:
        
        ```
        P[Default] = N(-d₂)
        
        where d₂ = [ln(V₀/D) + (r - ½σ²)T] / (σ√T)
        ```
        
        **Key Insights from the Sensitivity Charts:**
        
        1. **Volatility Impact (Top Left)**: Higher asset volatility increases default probability exponentially. This reflects greater uncertainty about future firm value.
        
        2. **Leverage Impact (Top Right)**: As debt-to-equity ratio increases, default probability rises sharply. Highly leveraged firms are closer to the default boundary.
        
        3. **Time Horizon Impact (Bottom Left)**: Longer time horizons generally increase default probability due to greater cumulative uncertainty, though the relationship can be non-monotonic.
        
        4. **Interest Rate Impact (Bottom Right)**: Higher risk-free rates reduce default probability by increasing the expected growth rate of assets.
        
        **Model Strengths:**
        - Economic foundation linking default to capital structure
        - Uses observable market data (equity prices and volatility)  
        - No-arbitrage pricing framework
        - Useful for hybrid instruments and capital structure decisions
        
        **Model Limitations:**
        - Default only at maturity (no early default)
        - Single debt issue assumption
        - Requires unobservable asset values
        - Poor short-term default prediction accuracy
        """)
    
    # === SECTION 3: PORTFOLIO CREDIT RISK ===
    st.subheader("📈 Portfolio Credit Risk & Loss Distributions")
    
    st.markdown("""
    **Portfolio Credit Risk** extends beyond individual default probabilities to consider correlation effects, 
    diversification benefits, and tail risk measures. This section demonstrates Monte Carlo simulation techniques 
    for generating portfolio loss distributions and calculating Value-at-Risk (VaR) and Conditional VaR (CVaR) metrics.
    """)
    
    # Portfolio Setup
    with st.container():
        st.markdown("#### Portfolio Configuration")
        
        portfolio_col1, portfolio_col2 = st.columns(2)
        
        with portfolio_col1:
            st.markdown("**Portfolio Parameters**")
            
            portfolio_type = st.selectbox("Portfolio Type", [
                "Corporate Lending", 
                "Sovereign Bonds", 
                "Mixed Credit Portfolio",
                "High Yield Focus",
                "Custom Portfolio"
            ])
            
            # Predefined portfolio configurations
            if portfolio_type == "Corporate Lending":
                n_obligors = 50
                avg_exposure = 10.0
                avg_pd = 0.025
                correlation = 0.15
                recovery = 0.40
            elif portfolio_type == "Sovereign Bonds":
                n_obligors = 25  
                avg_exposure = 20.0
                avg_pd = 0.015
                correlation = 0.25
                recovery = 0.60
            elif portfolio_type == "Mixed Credit Portfolio":
                n_obligors = 100
                avg_exposure = 5.0
                avg_pd = 0.03
                correlation = 0.12
                recovery = 0.45
            elif portfolio_type == "High Yield Focus":
                n_obligors = 30
                avg_exposure = 15.0
                avg_pd = 0.08
                correlation = 0.20
                recovery = 0.30
            else:  # Custom
                n_obligors = st.slider("Number of Obligors", 10, 200, 50)
                avg_exposure = st.number_input("Average Exposure ($M)", 1.0, 100.0, 10.0)
                avg_pd = st.slider("Average PD (%)", 0.5, 15.0, 2.5) / 100
                correlation = st.slider("Asset Correlation", 0.0, 0.5, 0.15, 0.05)
                recovery = st.slider("Recovery Rate (%)", 20.0, 80.0, 40.0) / 100
        
        with portfolio_col2:
            st.markdown("**Simulation Parameters**") 
            
            confidence_level = st.selectbox("VaR Confidence Level", [95, 99, 99.9])
            n_simulations = st.selectbox("Monte Carlo Simulations", [1000, 5000, 10000], index=2)
            
            # Generate portfolio
            if st.button("🎲 Generate Portfolio & Run Simulation", use_container_width=True):
                # Create random portfolio
                np.random.seed(42)  # For reproducibility
                
                # Generate exposures (log-normal distribution)
                exposures = np.random.lognormal(np.log(avg_exposure), 0.3, n_obligors)
                
                # Generate PDs with some dispersion
                pds = np.random.beta(2, 50/avg_pd - 2, n_obligors) if avg_pd < 0.5 else np.random.uniform(0.01, avg_pd*2, n_obligors)
                pds = np.clip(pds, 0.001, 0.25)  # Reasonable bounds
                
                # Uniform recovery rates for simplicity  
                recovery_rates = np.full(n_obligors, recovery)
                
                # Store in session state
                st.session_state.portfolio_data = {
                    'exposures': exposures,
                    'pds': pds,
                    'recovery_rates': recovery_rates,
                    'correlation': correlation,
                    'n_simulations': n_simulations,
                    'confidence_level': confidence_level
                }
                
                st.success(f"Generated portfolio with {n_obligors} obligors, total exposure ${np.sum(exposures):.1f}M")
    
    # Portfolio Analysis Results
    if 'portfolio_data' in st.session_state:
        portfolio = st.session_state.portfolio_data
        
        # Run Monte Carlo simulation
        with st.spinner("Running Monte Carlo simulation..."):
            losses = simulate_credit_portfolio_losses(
                len(portfolio['exposures']),
                portfolio['pds'],
                portfolio['correlation'],
                portfolio['recovery_rates'],
                portfolio['exposures'],
                portfolio['n_simulations']
            )
        
        # Calculate risk metrics
        expected_loss = np.mean(losses)
        unexpected_loss = np.std(losses)
        var_level = portfolio['confidence_level'] / 100
        var = np.percentile(losses, var_level * 100)
        cvar = np.mean(losses[losses >= var])
        
        # Display results
        st.markdown("#### Portfolio Risk Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("Expected Loss", f"${expected_loss:.2f}M")
        metrics_col2.metric("Unexpected Loss (Std)", f"${unexpected_loss:.2f}M") 
        metrics_col3.metric(f"VaR ({portfolio['confidence_level']}%)", f"${var:.2f}M")
        metrics_col4.metric(f"CVaR ({portfolio['confidence_level']}%)", f"${cvar:.2f}M")
        
        # Loss Distribution Chart
        st.markdown("#### Portfolio Loss Distribution")
        
        fig_loss_dist = go.Figure()
        
        # Histogram of losses
        fig_loss_dist.add_trace(go.Histogram(
            x=losses,
            nbinsx=50,
            opacity=0.7,
            name="Loss Distribution",
            marker_color='lightblue'
        ))
        
        # Add VaR and CVaR lines
        fig_loss_dist.add_vline(x=var, line_dash="dash", line_color="red", 
                               annotation_text=f"VaR ({portfolio['confidence_level']}%): ${var:.1f}M")
        fig_loss_dist.add_vline(x=cvar, line_dash="solid", line_color="darkred",
                               annotation_text=f"CVaR: ${cvar:.1f}M")
        fig_loss_dist.add_vline(x=expected_loss, line_dash="dot", line_color="green",
                               annotation_text=f"Expected Loss: ${expected_loss:.1f}M")
        
        fig_loss_dist.update_layout(
            title="Monte Carlo Portfolio Loss Distribution",
            xaxis_title="Portfolio Loss ($M)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_loss_dist, use_container_width=True)
        
        # Risk Contributions
        st.markdown("#### Risk Contribution Analysis")
        
        ul_contrib = calculate_unexpected_loss_contributions(
            portfolio['exposures'], 
            portfolio['pds'], 
            portfolio['correlation'], 
            portfolio['recovery_rates']
        )
        
        # Create DataFrame for top contributors
        contrib_df = pd.DataFrame({
            'Obligor': [f"Obligor {i+1}" for i in range(len(portfolio['exposures']))],
            'Exposure ($M)': portfolio['exposures'],
            'PD (%)': portfolio['pds'] * 100,
            'Individual UL ($M)': ul_contrib['individual_ul'],
            'Marginal Contrib': ul_contrib['marginal_contributions'],
            'Component UL ($M)': ul_contrib['component_contributions']
        })
        
        # Sort by component contribution and show top 10
        contrib_df_sorted = contrib_df.nlargest(10, 'Component UL ($M)')
        
        # Display top contributors
        st.dataframe(contrib_df_sorted.round(3), use_container_width=True)
        
        # Risk contribution pie chart
        fig_contrib = go.Figure(data=[go.Pie(
            labels=contrib_df_sorted['Obligor'],
            values=contrib_df_sorted['Component UL ($M)'],
            hole=0.3
        )])
        
        fig_contrib.update_layout(
            title="Top 10 Risk Contributors (Unexpected Loss)",
            height=400
        )
        
        st.plotly_chart(fig_contrib, use_container_width=True)
    
    with st.expander("📖 Understanding Portfolio Credit Risk"):
        st.markdown("""
        ### Portfolio Risk Modeling Concepts
        
        **Monte Carlo Simulation Process:**
        
        1. **Factor Model Structure**: Each obligor's creditworthiness is driven by:
           - Common systematic factor (economic conditions)  
           - Idiosyncratic factor (firm-specific risk)
        
        2. **Default Correlation**: Asset correlation parameter controls how defaults cluster:
           - Low correlation (~5-10%): Defaults are mostly independent
           - Moderate correlation (~15-25%): Some clustering during stress
           - High correlation (~30%+): Significant default clustering
        
        3. **Loss Calculation**: For each simulation:
           ```
           Portfolio Loss = Σ(Default_i × LGD_i × Exposure_i)
           ```
        
        **Key Risk Metrics:**
        
        - **Expected Loss (EL)**: Average loss across all scenarios
        - **Unexpected Loss (UL)**: Standard deviation of loss distribution  
        - **Value at Risk (VaR)**: Maximum loss at specified confidence level
        - **Conditional VaR (CVaR)**: Average loss in worst-case scenarios beyond VaR
        
        **Risk Contribution Analysis:**
        
        - **Individual UL**: Stand-alone unexpected loss for each obligor
        - **Marginal Contribution**: Change in portfolio UL from small exposure increase
        - **Component Contribution**: Portion of total portfolio UL attributed to each obligor
        
        **Interpretation Guidelines:**
        
        - **VaR vs CVaR**: CVaR always exceeds VaR and provides insight into tail severity
        - **Concentration Risk**: Top contributors often drive disproportionate portfolio risk
        - **Correlation Impact**: Higher correlation increases tail risk exponentially
        - **Diversification Benefits**: Diminish as correlation increases, especially in stress scenarios
        """)
    
    # === SECTION 4: CREDIT RATING TRANSITIONS & MIGRATION RISK ===
    st.subheader("📊 Credit Rating Transitions & Migration Risk")
    
    st.markdown("""
    **Credit Migration Risk** captures the impact of rating changes on portfolio value. Even without defaults, 
    downgrades can cause significant mark-to-market losses, trigger regulatory breaches, or force asset sales. 
    This section demonstrates how transition matrices are used to model migration risk and calculate Credit VaR.
    """)
    
    # Rating Transition Matrix
    with st.container():
        st.markdown("#### Historical Rating Transition Matrix")
        
        transition_matrix = create_rating_transition_matrix()
        
        # Display the transition matrix with formatting
        st.dataframe(
            transition_matrix.style.format("{:.2f}%").background_gradient(
                cmap='RdYlBu_r', subset=transition_matrix.columns[:-1]
            ),
            use_container_width=True
        )
        
        st.markdown("""
        **How to Read the Matrix:**
        - Each row shows the probability of migrating FROM that rating
        - Each column shows the probability of migrating TO that rating  
        - Diagonal elements (staying in same rating) are typically highest
        - Default probabilities are in the rightmost column
        """)
    
    # Migration Impact Analysis
    migration_col1, migration_col2 = st.columns(2)
    
    with migration_col1:
        st.markdown("#### Migration Scenario Analysis")
        
        current_rating = st.selectbox("Current Rating", transition_matrix.index[:-1])
        bond_value = st.number_input("Bond Value ($M)", value=100.0, min_value=1.0)
        time_horizon_mig = st.selectbox("Time Horizon", ["1 Year", "2 Years", "3 Years"])
        
        # Rating spread mappings (simplified)
        rating_spreads = {
            'AAA': 50, 'AA': 80, 'A': 120, 'BBB': 180, 
            'BB': 400, 'B': 700, 'CCC': 1200, 'Default': 10000
        }
        
        duration = st.slider("Modified Duration", 1.0, 10.0, 5.0, 0.5)
        
        # Calculate migration impact
        current_spread = rating_spreads[current_rating]
        migration_scenarios = []
        
        for target_rating in transition_matrix.columns:
            prob = transition_matrix.loc[current_rating, target_rating] / 100
            target_spread = rating_spreads[target_rating]
            spread_change = target_spread - current_spread
            price_impact = -duration * spread_change / 10000  # Duration × spread change
            value_change = bond_value * price_impact / 100
            
            migration_scenarios.append({
                'Target Rating': target_rating,
                'Probability (%)': prob * 100,
                'Spread Change (bps)': spread_change,
                'Price Impact (%)': price_impact,
                'Value Change ($M)': value_change
            })
        
        migration_df = pd.DataFrame(migration_scenarios)
        
    with migration_col2:
        st.markdown("#### Expected Migration Impact")
        
        st.dataframe(migration_df.round(3), use_container_width=True)
        
        # Calculate expected loss from migration
        expected_migration_loss = np.sum(migration_df['Probability (%)'] * migration_df['Value Change ($M)']) / 100
        st.metric("Expected Value Change", f"${expected_migration_loss:.2f}M")
        
        # Migration probability chart
        fig_migration = go.Figure(data=[go.Bar(
            x=migration_df['Target Rating'],
            y=migration_df['Probability (%)'],
            marker_color=['red' if x == 'Default' else 'orange' if 'C' in x else 'yellow' if x in ['BB', 'B'] else 'green' 
                         for x in migration_df['Target Rating']]
        )])
        
        fig_migration.update_layout(
            title=f"Migration Probabilities from {current_rating}",
            xaxis_title="Target Rating",
            yaxis_title="Probability (%)",
            height=300
        )
        
        st.plotly_chart(fig_migration, use_container_width=True)
    
    with st.expander("📖 Understanding Credit Migration Risk"):
        st.markdown("""
        ### Credit Migration Modeling Framework
        
        **Transition Matrix Construction:**
        - Built from historical rating agency data (Moody's, S&P, Fitch)
        - Typically covers 1-year horizons but can be extended using matrix multiplication
        - Shows higher stability for investment grade vs high yield ratings
        - Default probabilities increase significantly for lower ratings
        
        **Key Migration Risk Patterns:**
        
        1. **Rating Momentum**: Downgrades often occur in clusters rather than single notches
        2. **Asymmetric Transitions**: Downgrades are more common than upgrades during stress
        3. **Boundary Effects**: BBB/BB boundary is critical for regulatory and mandate constraints
        4. **Recovery Patterns**: Upgrades from distressed ratings are relatively rare
        
        **Economic Impact of Migrations:**
        
        - **Spread Widening**: Credit spreads increase exponentially with rating deterioration
        - **Duration Risk**: Longer duration bonds have higher sensitivity to spread changes
        - **Liquidity Impact**: Lower-rated bonds often become less liquid, widening bid-ask spreads
        - **Regulatory Consequences**: Downgrades can trigger capital charges or forced sales
        
        **Risk Management Applications:**
        
        - **Credit VaR**: Incorporates both default and migration losses
        - **Stress Testing**: Models rating migrations under adverse scenarios
        - **Portfolio Optimization**: Considers migration risk in asset allocation
        - **Hedging**: Uses CDS to hedge against rating deterioration risk
        """)
    
    # === SECTION 5: REGULATORY CAPITAL REQUIREMENTS ===
    st.subheader("🏛️ Basel III/IV Regulatory Capital Framework")
    
    st.markdown("""
    **Regulatory Capital Requirements** under Basel III/IV ensure banks maintain adequate capital buffers 
    to absorb credit losses and maintain financial stability. This section covers the key capital ratios, 
    risk-weighted asset calculations, and stress testing requirements.
    """)
    
    # Capital Ratio Calculator
    with st.container():
        st.markdown("#### Capital Adequacy Assessment")
        
        capital_col1, capital_col2 = st.columns(2)
        
        with capital_col1:
            st.markdown("**Bank Balance Sheet Information**")
            
            bank_type = st.selectbox("Bank Type", [
                "Large International Bank",
                "Regional Bank", 
                "Community Bank",
                "Custom Bank"
            ])
            
            # Predefined bank profiles
            if bank_type == "Large International Bank":
                total_assets = 500000  # $500B
                tier1_capital = 45000  # $45B
                tier2_capital = 10000  # $10B
                rwa = 350000  # $350B
            elif bank_type == "Regional Bank":
                total_assets = 100000  # $100B
                tier1_capital = 12000  # $12B
                tier2_capital = 3000   # $3B
                rwa = 75000   # $75B
            elif bank_type == "Community Bank":
                total_assets = 10000   # $10B
                tier1_capital = 1200   # $1.2B
                tier2_capital = 300    # $300M
                rwa = 7500    # $7.5B
            else:  # Custom
                total_assets = st.number_input("Total Assets ($M)", value=100000, min_value=1000, step=1000)
                tier1_capital = st.number_input("Tier 1 Capital ($M)", value=12000, min_value=100, step=100)
                tier2_capital = st.number_input("Tier 2 Capital ($M)", value=3000, min_value=0, step=100)
                rwa = st.number_input("Risk-Weighted Assets ($M)", value=75000, min_value=500, step=500)
        
        with capital_col2:
            st.markdown("**Regulatory Capital Ratios**")
            
            # Calculate ratios
            ratios = calculate_regulatory_capital_ratios(total_assets, tier1_capital, tier2_capital, rwa)
            
            # Display ratios with regulatory thresholds
            ratio_col1, ratio_col2 = st.columns(2)
            
            ratio_col1.metric("CET1 Ratio", f"{ratios['CET1_Ratio']:.1f}%", 
                             help="Minimum: 4.5%, Buffer: 7.0%+")
            ratio_col1.metric("Total Capital Ratio", f"{ratios['Total_Capital_Ratio']:.1f}%",
                             help="Minimum: 8.0%, Well-capitalized: 10.0%+")
            
            ratio_col2.metric("Tier 1 Capital Ratio", f"{ratios['Tier1_Ratio']:.1f}%",
                             help="Minimum: 6.0%, Well-capitalized: 8.0%+")
            ratio_col2.metric("Leverage Ratio", f"{ratios['Leverage_Ratio']:.1f}%",
                             help="Minimum: 3.0%, Buffer: 4.0%+")
            
            # Capital adequacy assessment
            st.markdown("**Capital Adequacy Status**")
            
            def assess_capital_adequacy(ratio_value, min_req, well_cap_req):
                if ratio_value >= well_cap_req:
                    return "🟢 Well-Capitalized"
                elif ratio_value >= min_req:
                    return "🟡 Adequately Capitalized"
                else:
                    return "🔴 Undercapitalized"
            
            cet1_status = assess_capital_adequacy(ratios['CET1_Ratio'], 4.5, 7.0)
            tier1_status = assess_capital_adequacy(ratios['Tier1_Ratio'], 6.0, 8.0)
            total_cap_status = assess_capital_adequacy(ratios['Total_Capital_Ratio'], 8.0, 10.0)
            leverage_status = assess_capital_adequacy(ratios['Leverage_Ratio'], 3.0, 4.0)
            
            st.write(f"**CET1:** {cet1_status}")
            st.write(f"**Tier 1:** {tier1_status}")
            st.write(f"**Total Capital:** {total_cap_status}")
            st.write(f"**Leverage:** {leverage_status}")
    
    # Risk-Weighted Assets Breakdown
    st.markdown("#### Risk-Weighted Assets Composition")
    
    rwa_col1, rwa_col2 = st.columns(2)
    
    with rwa_col1:
        st.markdown("**RWA by Asset Class**")
        
        # Sample RWA composition
        rwa_composition = {
            'Corporate Lending': 0.35,
            'Retail Mortgages': 0.25,
            'Trading Book': 0.15,
            'Sovereign Exposures': 0.10,
            'Operational Risk': 0.10,
            'Other': 0.05
        }
        
        # Risk weights by asset class
        risk_weights = {
            'Corporate Lending': 75,    # 75% average risk weight
            'Retail Mortgages': 35,     # 35% risk weight
            'Trading Book': 150,        # 150% high risk
            'Sovereign Exposures': 20,  # 20% risk weight
            'Operational Risk': 125,    # 125% operational risk charge
            'Other': 100               # 100% standard risk weight
        }
        
        rwa_breakdown = []
        for asset_class, proportion in rwa_composition.items():
            rwa_amount = rwa * proportion
            implied_exposure = rwa_amount / (risk_weights[asset_class] / 100)
            
            rwa_breakdown.append({
                'Asset Class': asset_class,
                'RWA ($M)': rwa_amount,
                'Risk Weight (%)': risk_weights[asset_class],
                'Implied Exposure ($M)': implied_exposure,
                'Share of RWA (%)': proportion * 100
            })
        
        rwa_df = pd.DataFrame(rwa_breakdown)
        st.dataframe(rwa_df.round(0), use_container_width=True)
    
    with rwa_col2:
        st.markdown("**RWA Distribution**")
        
        fig_rwa = go.Figure(data=[go.Pie(
            labels=list(rwa_composition.keys()),
            values=list(rwa_composition.values()),
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig_rwa.update_layout(
            title="Risk-Weighted Assets by Category",
            height=400
        )
        
        st.plotly_chart(fig_rwa, use_container_width=True)
    
    # Stress Testing Framework
    st.markdown("#### Stress Testing & CCAR Analysis")
    
    stress_col1, stress_col2 = st.columns(2)
    
    with stress_col1:
        st.markdown("**Stress Test Scenarios**")
        
        # Define stress scenarios
        stress_scenarios = {
            'Baseline': {'GDP_shock': 0.0, 'unemployment_shock': 0.0, 'house_price_shock': 0.0},
            'Adverse': {'GDP_shock': -3.0, 'unemployment_shock': 4.0, 'house_price_shock': -15.0},
            'Severely Adverse': {'GDP_shock': -6.5, 'unemployment_shock': 8.5, 'house_price_shock': -25.0}
        }
        
        selected_scenario = st.selectbox("Stress Scenario", list(stress_scenarios.keys()))
        scenario = stress_scenarios[selected_scenario]
        
        st.write(f"**{selected_scenario} Scenario:**")
        st.write(f"- GDP Shock: {scenario['GDP_shock']:+.1f}%")
        st.write(f"- Unemployment Shock: {scenario['unemployment_shock']:+.1f}pp")
        st.write(f"- House Price Shock: {scenario['house_price_shock']:+.1f}%")
        
        # Calculate stress impact on credit losses
        base_loss_rate = 0.01  # 1% base loss rate
        
        # Simple stress multipliers (in practice, much more complex)
        gdp_sensitivity = -0.3      # 30bp increase per 1% GDP decline
        unemp_sensitivity = 0.2     # 20bp increase per 1pp unemployment increase
        house_sensitivity = -0.1    # 10bp increase per 1% house price decline
        
        stressed_loss_rate = base_loss_rate + (
            scenario['GDP_shock'] * gdp_sensitivity / 100 +
            scenario['unemployment_shock'] * unemp_sensitivity / 100 +
            scenario['house_price_shock'] * house_sensitivity / 100
        )
        
        stressed_losses = rwa * stressed_loss_rate
        
    with stress_col2:
        st.markdown("**Stress Test Results**")
        
        st.metric("Base Loss Rate", f"{base_loss_rate:.2%}")
        st.metric("Stressed Loss Rate", f"{stressed_loss_rate:.2%}", 
                 f"{(stressed_loss_rate - base_loss_rate):.2%}")
        st.metric("Projected Credit Losses", f"${stressed_losses:.0f}M")
        
        # Impact on capital ratios
        post_stress_tier1 = tier1_capital - stressed_losses
        post_stress_cet1_ratio = (post_stress_tier1 / rwa) * 100
        
        st.metric("Post-Stress CET1 Ratio", f"{post_stress_cet1_ratio:.1f}%",
                 f"{post_stress_cet1_ratio - ratios['CET1_Ratio']:.1f}pp")
        
        # Capital adequacy post-stress
        if post_stress_cet1_ratio >= 7.0:
            stress_status = "🟢 Passes Stress Test"
        elif post_stress_cet1_ratio >= 4.5:
            stress_status = "🟡 Marginally Adequate"
        else:
            stress_status = "🔴 Fails Stress Test"
            
        st.write(f"**Stress Test Result:** {stress_status}")
        
        # Stress test chart
        scenarios_chart = list(stress_scenarios.keys())
        cet1_ratios_stressed = []
        
        for scenario_name in scenarios_chart:
            scen = stress_scenarios[scenario_name]
            stressed_lr = base_loss_rate + (
                scen['GDP_shock'] * gdp_sensitivity / 100 +
                scen['unemployment_shock'] * unemp_sensitivity / 100 +
                scen['house_price_shock'] * house_sensitivity / 100
            )
            stressed_loss = rwa * stressed_lr
            post_stress_t1 = tier1_capital - stressed_loss
            post_stress_ratio = (post_stress_t1 / rwa) * 100
            cet1_ratios_stressed.append(post_stress_ratio)
        
        fig_stress = go.Figure(data=[go.Bar(
            x=scenarios_chart,
            y=cet1_ratios_stressed,
            marker_color=['green', 'orange', 'red']
        )])
        
        fig_stress.add_hline(y=4.5, line_dash="dash", line_color="red", 
                            annotation_text="Minimum CET1 (4.5%)")
        fig_stress.add_hline(y=7.0, line_dash="dash", line_color="blue",
                            annotation_text="Well-Capitalized (7.0%)")
        
        fig_stress.update_layout(
            title="CET1 Ratios Under Stress Scenarios",
            xaxis_title="Scenario",
            yaxis_title="CET1 Ratio (%)",
            height=400
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)
    
    with st.expander("📖 Understanding Regulatory Capital Requirements"):
        st.markdown("""
        ### Basel III/IV Capital Framework
        
        **Capital Components:**
        
        1. **Common Equity Tier 1 (CET1)**:
           - Highest quality capital (common stock + retained earnings)
           - Primary loss-absorbing capacity
           - Minimum requirement: 4.5% + conservation buffer (2.5%) = 7.0%
        
        2. **Tier 1 Capital**:
           - CET1 + Additional Tier 1 instruments (contingent convertible bonds)
           - Minimum requirement: 6.0% + conservation buffer = 8.5%
        
        3. **Total Capital**:
           - Tier 1 + Tier 2 capital (subordinated debt, loan loss provisions)
           - Minimum requirement: 8.0% + conservation buffer = 10.5%
        
        4. **Leverage Ratio**:
           - Tier 1 capital / Total exposure (not risk-weighted)
           - Minimum requirement: 3.0% (4.0% for G-SIBs)
        
        **Risk-Weighted Assets Calculation:**
        
        ```
        RWA = Σ(Exposure_i × Risk_Weight_i)
        ```
        
        **Standard Risk Weights:**
        - Sovereigns (OECD): 0-20%
        - Bank exposures: 20-150%
        - Corporate exposures: 75-150%
        - Retail exposures: 75%
        - Residential mortgages: 35-100%
        - Commercial real estate: 100%+
        
        **Stress Testing Requirements:**
        
        1. **CCAR/DFAST (US)**: Annual comprehensive stress tests
        2. **EU Stress Tests**: EBA-coordinated biannual exercises
        3. **Internal Stress Tests**: Bank-specific scenario analysis
        
        **Key Stress Test Elements:**
        
        - **Macroeconomic Scenarios**: GDP, unemployment, interest rates, asset prices
        - **Credit Loss Modeling**: Loan loss provisions and charge-offs
        - **Revenue Impact**: Net interest income and fee income stress
        - **Capital Actions**: Dividend and share repurchase constraints
        - **Recovery Planning**: Actions to restore capital adequacy
        
        **Regulatory Buffers:**
        
        - **Capital Conservation Buffer**: 2.5% of RWA
        - **Countercyclical Capital Buffer**: 0-2.5% (varies by jurisdiction)
        - **G-SIB Buffer**: 1-3.5% for globally systemically important banks
        - **D-SIB Buffer**: 0-2% for domestically systemically important banks
        """)
    
    # === SECTION 6: PRACTICAL APPLICATIONS ===
    st.subheader("💼 Practical Applications & Case Studies")
    
    applications_tabs = st.tabs(["🏦 Bank Portfolio Management", "📊 Credit Derivatives Trading", "🏛️ Regulatory Compliance"])
    
    with applications_tabs[0]:
        st.markdown("""
        ### Bank Portfolio Management Applications
        
        **1. Loan Portfolio Optimization**
        - Use Credit VaR models to set concentration limits
        - Optimize portfolio composition to maximize risk-adjusted returns
        - Balance expected returns against unexpected loss and regulatory capital
        
        **2. Pricing Credit Products**  
        - Incorporate individual and portfolio risk metrics into loan pricing
        - Account for correlation benefits in portfolio pricing
        - Set risk-adjusted hurdle rates using RAROC (Risk-Adjusted Return on Capital)
        
        **3. Capital Allocation**
        - Allocate economic capital based on marginal risk contributions
        - Guide business line strategy based on capital efficiency
        - Support M&A decisions with portfolio risk analysis
        
        **4. Hedging Strategies**
        - Use Credit Default Swaps to hedge single-name exposures
        - Hedge sector concentrations with credit indices (CDX, iTraxx)
        - Implement dynamic hedging based on correlation changes
        
        **Case Study: Regional Bank Credit Portfolio**
        
        A regional bank with $50B in loans uses Credit VaR modeling to:
        - Identify that 40% of portfolio risk comes from 10% of exposures
        - Discover high correlation between real estate and energy sectors
        - Implement sector limits reducing tail risk by 25%
        - Hedge top 10 single-name exposures with CDS, freeing up $500M in regulatory capital
        """)
        
        # Interactive case study
        st.markdown("#### Interactive Case Study: Portfolio Rebalancing")
        
        case_col1, case_col2 = st.columns(2)
        
        with case_col1:
            st.markdown("**Current Portfolio Composition**")
            
            sectors = ['Technology', 'Healthcare', 'Energy', 'Real Estate', 'Manufacturing']
            current_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
            sector_correlations = [0.15, 0.12, 0.25, 0.30, 0.18]
            sector_pd = [0.02, 0.015, 0.05, 0.04, 0.03]
            
            # Allow user to adjust weights
            st.markdown("**Adjust Portfolio Weights:**")
            new_weights = []
            for i, sector in enumerate(sectors):
                weight = st.slider(f"{sector} (%)", 0, 50, int(current_weights[i]*100)) / 100
                new_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(new_weights)
            if total_weight > 0:
                new_weights = [w/total_weight for w in new_weights]
            else:
                new_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        with case_col2:
            st.markdown("**Portfolio Risk Analysis**")
            
            # Calculate portfolio risk metrics
            portfolio_pd = sum(w * pd for w, pd in zip(new_weights, sector_pd))
            portfolio_correlation = sum(w * corr for w, corr in zip(new_weights, sector_correlations))
            
            # Risk concentration (Herfindahl index)
            concentration_risk = sum(w**2 for w in new_weights)
            
            st.metric("Portfolio PD", f"{portfolio_pd:.2%}")
            st.metric("Avg Correlation", f"{portfolio_correlation:.2%}")
            st.metric("Concentration Risk", f"{concentration_risk:.3f}", help="Lower is better (0.2 = perfectly diversified)")
            
            # Risk contribution chart
            risk_contributions = [w * pd * corr for w, pd, corr in zip(new_weights, sector_pd, sector_correlations)]
            total_risk = sum(risk_contributions)
            
            if total_risk > 0:
                risk_pcts = [rc/total_risk for rc in risk_contributions]
                
                fig_risk_contrib = go.Figure(data=[go.Bar(
                    x=sectors,
                    y=[rp*100 for rp in risk_pcts],
                    marker_color=['red' if rp > 0.3 else 'orange' if rp > 0.2 else 'green' for rp in risk_pcts]
                )])
                
                fig_risk_contrib.update_layout(
                    title="Risk Contribution by Sector",
                    xaxis_title="Sector", 
                    yaxis_title="Risk Contribution (%)",
                    height=300
                )
                
                st.plotly_chart(fig_risk_contrib, use_container_width=True)
    
    with applications_tabs[1]:
        st.markdown("""
        ### Credit Derivatives Trading Applications
        
        **1. Single-Name CDS Trading**
        - Express views on individual credit quality  
        - Hedge specific exposures without selling underlying bonds
        - Arbitrage between bond and CDS prices (basis trading)
        - Capital relief for regulatory purposes
        
        **2. Index Trading (CDX/iTraxx)**  
        - Macro hedge for broad credit exposure
        - Express views on credit spread direction
        - Trade credit volatility through options on indices
        - Relative value between different credit indices
        
        **3. Tranched Credit Products**
        - Trade correlation views through CDO tranches
        - Tailor risk/return profiles for different investor needs
        - Benefit from diversification in portfolio products
        - Manage concentration risk in structured products
        
        **4. Basis and Relative Value Strategies**
        - CDS-Bond basis: Trade differences between CDS and cash bond spreads
        - Capital structure arbitrage: Long/short different parts of capital stack
        - Curve trades: Express views on credit curve shape and evolution
        - Cross-currency basis: Exploit pricing differences across currencies
        
        **Trading Strategy Example: Credit Curve Flattening**
        
        A hedge fund identifies that the 1Y-5Y credit curve for a technology company is unusually steep:
        - Current 1Y CDS: 150 bps
        - Current 5Y CDS: 250 bps  
        - Historical 1Y-5Y spread: 60 bps vs current 100 bps
        
        **Trade Structure:**
        - Buy protection on 1Y CDS (pay 150 bps)
        - Sell protection on 5Y CDS (receive 250 bps)
        - Net receive 100 bps annually
        
        **Profit Scenarios:**
        - Credit quality improves: Both spreads tighten, but curve flattens
        - Credit deteriorates gradually: Curve flattens as 1Y approaches 5Y level
        - No credit change: Earn carry from steep curve position
        
        **Risk Management:**
        - Delta hedge: Adjust notionals to maintain risk-neutral position
        - Correlation monitoring: Track relationship between different maturities
        - Liquidity management: Ensure ability to exit positions in stress
        """)
    
    with applications_tabs[2]:
        st.markdown("""
        ### Regulatory Compliance Applications
        
        **1. Basel III/IV Capital Planning**
        - Calculate minimum capital requirements across risk categories
        - Plan capital actions (retained earnings, issuance, distributions)
        - Optimize capital ratios while maintaining business objectives
        - Prepare for regulatory examinations and stress tests
        
        **2. IFRS 9 / CECL Expected Credit Loss**
        - Model lifetime expected losses for financial reporting
        - Incorporate forward-looking macroeconomic information
        - Validate models and assumptions for audit requirements
        - Manage earnings volatility from loss provision changes
        
        **3. Stress Testing Compliance**
        - Develop comprehensive stress testing frameworks
        - Submit CCAR/DFAST results to regulators
        - Conduct internal stress tests for risk management
        - Maintain governance and model validation processes
        
        **4. Large Exposure Rules**
        - Monitor single obligor concentration limits
        - Aggregate exposures across related entities
        - Apply appropriate exposure calculations and exemptions
        - Report large exposures to supervisors
        
        **Regulatory Timeline Example: CCAR Process**
        
        **January-March: Scenario Design**
        - Receive Federal Reserve scenarios
        - Develop institution-specific scenarios
        - Validate macroeconomic models and assumptions
        
        **April-June: Model Execution**
        - Run credit loss models across all portfolios
        - Calculate pre-provision net revenue projections  
        - Estimate regulatory capital ratios under stress
        - Perform model validation and sensitivity analysis
        
        **July-September: Results & Planning**
        - Submit results to Federal Reserve
        - Receive feedback and objections (if any)
        - Plan capital actions based on stress test results
        - Implement risk management improvements
        
        **October-December: Implementation**
        - Execute approved capital plans
        - Monitor actual performance vs. stress projections
        - Update models based on new data and regulations
        - Prepare for next year's stress testing cycle
        
        **Key Regulatory Ratios Monitoring:**
        - CET1 ratio maintained above 7% (including conservation buffer)
        - Leverage ratio above 4% for G-SIBs, 3% for others  
        - Total loss-absorbing capacity (TLAC) for G-SIBs: 18% of RWA
        - Liquidity coverage ratio (LCR) above 100%
        - Net stable funding ratio (NSFR) above 100%
        """)
    
    # === FOOTER & REFERENCES ===
    st.markdown("---")
    st.subheader("📚 Educational Summary & Key Takeaways")
    
    with st.expander("🎯 Key Learning Points", expanded=True):
        st.markdown("""
        ### Essential Credit Risk Management Concepts
        
        **1. Model Integration:**
        - Structural models (Merton) provide economic intuition but require unobservable inputs
        - Reduced-form models offer market calibration but lack fundamental linkage
        - Portfolio models must account for correlation to avoid underestimating tail risk
        - Regulatory models balance sophistication with standardization requirements
        
        **2. Risk Measurement Hierarchy:**
        - Individual obligor risk: PD, LGD, EAD (Probability of Default, Loss Given Default, Exposure at Default)
        - Portfolio risk: Diversification benefits, correlation effects, concentration penalties
        - Economic capital: VaR and CVaR for internal risk management and pricing
        - Regulatory capital: Standardized requirements for supervisory oversight
        
        **3. Critical Success Factors:**
        - Data quality and availability drive model accuracy and reliability
        - Model validation must be independent, comprehensive, and ongoing
        - Stress testing reveals model limitations under adverse conditions
        - Governance ensures models are used appropriately and limitations understood
        
        **4. Practical Implementation:**
        - Start with simple models and add complexity based on portfolio characteristics
        - Focus on major risk drivers rather than model sophistication for its own sake
        - Maintain model inventory and regular backtesting against actual outcomes
        - Train users on model assumptions, limitations, and appropriate applications
        
        **5. Regulatory Evolution:**
        - Basel framework continues evolving toward risk-sensitive approaches
        - Standardized approaches provide floors under internal model outputs  
        - Stress testing becomes more central to capital planning and risk management
        - Climate risk and operational resilience gain regulatory attention
        """)
    
    st.markdown("---")
    
    # Technical References
    with st.expander("📖 Technical References & Further Reading"):
        st.markdown("""
        ### Academic & Professional References
        
        **Foundational Texts:**
        1. Duffie, D. & Singleton, K. (2003). *Credit Risk: Pricing, Measurement, and Management*. Princeton University Press.
        2. Lando, D. (2004). *Credit Risk Modeling: Theory and Applications*. Princeton University Press.
        3. Schönbucher, P. (2003). *Credit Derivatives Pricing Models*. John Wiley & Sons.
        4. Bluhm, C., Overbeck, L. & Wagner, C. (2010). *Introduction to Credit Risk Modeling*. Chapman & Hall.
        
        **Regulatory Guidance:**
        - Basel Committee on Banking Supervision (2017). *Basel III: Finalising post-crisis reforms*
        - Federal Reserve SR 11-7: *Guidance on Model Risk Management*
        - ECB Guide to Internal Models (2018)
        - Bank of England SS1/18: *Model Risk Management Principles for Banks*
        
        **Industry Standards:**
        - ISDA Credit Derivatives Definitions (2014)
        - CreditMetrics Technical Document (1997) - J.P. Morgan
        - Moody's KMV Portfolio Manager Documentation
        - S&P Credit Analytics and Research
        
        **Key Academic Papers:**
        - Merton, R.C. (1974). "On the Pricing of Corporate Debt." *Journal of Finance*, 29(2), 449-470.
        - Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." *Risk*, 15(12), 160-162.
        - Li, D.X. (2000). "On Default Correlation: A Copula Function Approach." *Journal of Fixed Income*, 9(4), 43-54.
        - Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *Journal of Finance*, 23(4), 589-609.
        """)
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        <p><strong>Credit Risk Management & Regulatory Capital Framework</strong></p>
        <p>Comprehensive educational platform for understanding credit risk modeling and capital management</p>
        <p><em>For educational purposes only - Not investment or regulatory advice</em></p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# Run the page
if __name__ == "__main__":
    show_credit_risk_management_page()
