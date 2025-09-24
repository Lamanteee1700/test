import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import bs_price, greeks

# --- Structured Product Functions ---
def barrier_option_price(S, K, T, r, sigma, barrier, option_type="call", barrier_type="knock-out"):
    """Simplified barrier option pricing using Monte Carlo"""
    n_sims = 10000
    dt = T/252  # Daily steps
    n_steps = int(T/dt)
    
    payoffs = []
    
    for _ in range(n_sims):
        S_path = [S]
        alive = True
        
        for _ in range(n_steps):
            z = np.random.normal()
            S_new = S_path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            S_path.append(S_new)
            
            # Check barrier condition
            if barrier_type == "knock-out" and ((S_new >= barrier and option_type == "call") or 
                                               (S_new <= barrier and option_type == "put")):
                alive = False
                break
            elif barrier_type == "knock-in" and ((S_new >= barrier and option_type == "call") or 
                                                (S_new <= barrier and option_type == "put")):
                alive = True
        
        # Calculate payoff
        final_price = S_path[-1]
        if barrier_type == "knock-out":
            if alive:
                if option_type == "call":
                    payoff = max(0, final_price - K)
                else:
                    payoff = max(0, K - final_price)
            else:
                payoff = 0
        else:  # knock-in
            if alive:
                if option_type == "call":
                    payoff = max(0, final_price - K)
                else:
                    payoff = max(0, K - final_price)
            else:
                payoff = 0
                
        payoffs.append(payoff)
    
    return np.exp(-r*T) * np.mean(payoffs)

def reverse_convertible_payoff(S_final, K, barrier, coupon_rate, notional=100):
    """Calculate reverse convertible payoff"""
    if S_final >= barrier:
        return notional + coupon_rate * notional  # Full principal + coupon
    else:
        shares_received = notional / K
        return shares_received * S_final + coupon_rate * notional  # Stock value + coupon

def autocallable_payoff(S_path, S0, observation_dates, autocall_barrier, coupon_rate, knock_in_barrier, notional=100):
    """Calculate autocallable note payoff"""
    for i, date in enumerate(observation_dates):
        S_obs = S_path[int(date * len(S_path))]
        if S_obs >= autocall_barrier:
            # Auto-called - return principal plus accumulated coupons
            return notional + coupon_rate * notional * (i + 1)
    
    # Not auto-called, check knock-in
    min_price = min(S_path)
    if min_price >= knock_in_barrier:
        return notional + coupon_rate * notional * len(observation_dates)  # Full protection
    else:
        # Knock-in occurred
        final_price = S_path[-1]
        return notional * (final_price / S0)  # Equity performance

def capital_protected_note_payoff(S_final, S0, participation_rate, protection_level=1.0, notional=100):
    """Calculate capital protected note payoff"""
    equity_return = (S_final - S0) / S0
    equity_component = max(0, participation_rate * equity_return * notional)
    protected_amount = protection_level * notional
    return protected_amount + equity_component

# --- Streamlit App ---
def show_structured_products_page():
    st.title("🏗️ Structured Products Builder & Analyzer")
    
    st.markdown("""
    Build and analyze structured products by combining bonds, options, and swaps.
    Understand how these instruments create customized risk/return profiles.
    """)
    
    # Product Selection
    product_type = st.selectbox(
        "Choose Structured Product Type:",
        ["Reverse Convertible", "Autocallable Note", "Capital Protected Note", "Barrier Options"]
    )
    
    # Common Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Parameters")
        S0 = st.number_input("Current Asset Price ($)", value=100.0, min_value=1.0)
        r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 3.0, 0.25) / 100
        sigma = st.slider("Volatility (%)", 5.0, 80.0, 20.0, 2.5) / 100
        T = st.slider("Time to Maturity (years)", 0.25, 5.0, 1.0, 0.25)
        
    with col2:
        st.subheader("Product Parameters")
        notional = st.number_input("Notional Amount ($)", value=100.0, min_value=1.0)
        
        if product_type in ["Reverse Convertible", "Autocallable Note"]:
            coupon_rate = st.slider("Annual Coupon Rate (%)", 0.0, 20.0, 8.0, 0.5) / 100
        
        if product_type == "Capital Protected Note":
            participation_rate = st.slider("Equity Participation Rate (%)", 0.0, 200.0, 100.0, 5.0) / 100
            protection_level = st.slider("Capital Protection Level (%)", 80.0, 100.0, 100.0, 5.0) / 100
    
    # Product-Specific Parameters and Analysis
    if product_type == "Reverse Convertible":
        st.subheader("Reverse Convertible Note")
        
        barrier = st.slider("Knock-in Barrier (% of initial)", 50.0, 95.0, 70.0, 5.0) / 100 * S0
        
        st.markdown(f"""
        **Structure:** High coupon ({coupon_rate:.1%}) but risk of receiving stock if price falls below barrier.
        - **Barrier Level:** ${barrier:.2f} ({barrier/S0:.0%} of initial price)
        - **Annual Coupon:** {coupon_rate:.1%}
        """)
        
        # Payoff Calculation
        price_range = np.linspace(0.3*S0, 1.5*S0, 100)
        payoffs = [reverse_convertible_payoff(S, S0, barrier, coupon_rate, notional) for S in price_range]
        
        # Compare with vanilla bond
        vanilla_bond_payoff = [notional + coupon_rate * notional] * len(price_range)
        
        # Risk Analysis
        prob_conversion = 1 - norm.cdf((np.log(barrier/S0) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
        
        st.metric("Probability of Stock Conversion", f"{prob_conversion:.1%}")
        
    elif product_type == "Autocallable Note":
        st.subheader("Autocallable Note")
        
        autocall_barrier = st.slider("Auto-call Barrier (% of initial)", 95.0, 120.0, 100.0, 5.0) / 100 * S0
        knock_in_barrier = st.slider("Knock-in Barrier (% of initial)", 50.0, 85.0, 65.0, 5.0) / 100 * S0
        n_observations = st.selectbox("Observation Frequency", [4, 2, 1], index=0, 
                                    format_func=lambda x: f"{'Quarterly' if x==4 else 'Semi-annual' if x==2 else 'Annual'}")
        
        observation_dates = np.linspace(1/n_observations, T, int(T*n_observations))
        
        st.markdown(f"""
        **Structure:** Potential early redemption with enhanced coupons, but downside risk if barriers breached.
        - **Auto-call Barrier:** ${autocall_barrier:.2f} ({autocall_barrier/S0:.0%})  
        - **Knock-in Barrier:** ${knock_in_barrier:.2f} ({knock_in_barrier/S0:.0%})
        - **Observations:** {len(observation_dates)} times per year
        """)
        
        # Monte Carlo simulation for autocallable
        n_sims = 1000
        payoffs_autocall = []
        
        for _ in range(n_sims):
            # Generate price path
            dt = T/252
            n_steps = int(T/dt)
            S_path = [S0]
            
            for _ in range(n_steps):
                z = np.random.normal()
                S_new = S_path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
                S_path.append(S_new)
            
            payoff = autocallable_payoff(S_path, S0, observation_dates, autocall_barrier, 
                                       coupon_rate, knock_in_barrier, notional)
            payoffs_autocall.append(payoff)
        
        expected_payoff = np.mean(payoffs_autocall)
        payoff_std = np.std(payoffs_autocall)
        
        st.metric("Expected Payoff", f"${expected_payoff:.2f}")
        st.metric("Payoff Standard Deviation", f"${payoff_std:.2f}")
        
        # Payoff distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=payoffs_autocall,
            nbinsx=30,
            name="Payoff Distribution",
            opacity=0.7
        ))
        fig_hist.add_vline(x=expected_payoff, line_dash="dash", line_color="red",
                          annotation_text=f"Expected: ${expected_payoff:.2f}")
        fig_hist.update_layout(title="Autocallable Note Payoff Distribution", 
                             xaxis_title="Payoff ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    elif product_type == "Capital Protected Note":
        st.subheader("Capital Protected Note")
        
        st.markdown(f"""
        **Structure:** Guaranteed return of {protection_level:.0%} capital plus {participation_rate:.0%} equity upside.
        - **Protection Level:** {protection_level:.1%} of notional
        - **Equity Participation:** {participation_rate:.1%}
        """)
        
        # Component Analysis
        bond_component = protection_level * notional * np.exp(-r * T)  # PV of protection
        option_budget = notional - bond_component
        
        # Calculate equivalent call option details
        call_price = bs_price(S0, S0, T, r, sigma, "call")  # ATM call
        equivalent_options = option_budget / call_price
        
        st.subheader("Product Decomposition")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        comp_col1.metric("Bond Component", f"${bond_component:.2f}")
        comp_col2.metric("Option Budget", f"${option_budget:.2f}")
        comp_col3.metric("Equivalent ATM Calls", f"{equivalent_options:.2f}")
        
        # Payoff diagram
        price_range = np.linspace(0.5*S0, 2.0*S0, 100)
        payoffs = [capital_protected_note_payoff(S, S0, participation_rate, protection_level, notional) 
                  for S in price_range]
        
    elif product_type == "Barrier Options":
        st.subheader("Barrier Options Analysis")
        
        barrier_level = st.slider("Barrier Level ($)", 0.5*S0, 1.5*S0, 0.8*S0, S0*0.05)
        barrier_type = st.selectbox("Barrier Type", ["knock-out", "knock-in"])
        option_type = st.selectbox("Option Type", ["call", "put"])
        K = st.number_input("Strike Price ($)", value=S0, min_value=1.0)
        
        # Price barrier option
        barrier_price = barrier_option_price(S0, K, T, r, sigma, barrier_level, option_type, barrier_type)
        vanilla_price = bs_price(S0, K, T, r, sigma, option_type)
        
        st.metric("Barrier Option Price", f"${barrier_price:.3f}")
        st.metric("Vanilla Option Price", f"${vanilla_price:.3f}")
        st.metric("Price Difference", f"${vanilla_price - barrier_price:.3f}")
        
        # Show barrier effect
        st.markdown(f"""
        **{barrier_type.title()} {option_type.title()} Option**
        - **Barrier:** ${barrier_level:.2f}
        - **Strike:** ${K:.2f}  
        - **Barrier Impact:** {((vanilla_price - barrier_price)/vanilla_price)*100:.1f}% price reduction
        """)
    
    # General Payoff Diagram (for non-autocallable products)
    if product_type != "Autocallable Note":
        st.subheader("Payoff Diagram")
        
        if product_type != "Barrier Options":
            fig = go.Figure()
            
            if product_type == "Reverse Convertible":
                fig.add_trace(go.Scatter(x=price_range, y=payoffs, mode='lines', 
                                       name='Reverse Convertible', line=dict(width=3, color='red')))
                fig.add_trace(go.Scatter(x=price_range, y=vanilla_bond_payoff, mode='lines', 
                                       name='Vanilla Bond', line=dict(width=2, dash='dash', color='blue')))
                fig.add_vline(x=barrier, line_dash="dot", line_color="orange", 
                            annotation_text=f"Barrier: ${barrier:.0f}")
                
            elif product_type == "Capital Protected Note":
                # Add vanilla equity for comparison
                equity_payoffs = [(S/S0) * notional for S in price_range]
                bond_payoffs = [protection_level * notional] * len(price_range)
                
                fig.add_trace(go.Scatter(x=price_range, y=payoffs, mode='lines', 
                                       name='Capital Protected Note', line=dict(width=3, color='green')))
                fig.add_trace(go.Scatter(x=price_range, y=equity_payoffs, mode='lines', 
                                       name='Direct Equity', line=dict(width=2, dash='dash', color='blue')))
                fig.add_trace(go.Scatter(x=price_range, y=bond_payoffs, mode='lines', 
                                       name='Protected Amount', line=dict(width=2, dash='dot', color='gray')))
            
            fig.add_vline(x=S0, line_dash="dash", line_color="black", annotation_text="Current Price")
            fig.update_layout(title=f"{product_type} Payoff Profile", 
                            xaxis_title="Asset Price at Maturity ($)", 
                            yaxis_title="Payoff ($)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Analysis Section
    st.subheader("Risk Analysis")
    
    # Sensitivity Analysis
    sensitivity_param = st.selectbox("Sensitivity Analysis Parameter", 
                                   ["Volatility", "Time to Maturity", "Interest Rate"])
    
    if sensitivity_param == "Volatility":
        vol_range = np.linspace(0.1, 0.6, 20)
        param_values = vol_range
        param_name = "Volatility"
        
    elif sensitivity_param == "Time to Maturity":
        time_range = np.linspace(0.25, 3.0, 20)
        param_values = time_range
        param_name = "Time to Maturity (years)"
        
    else:  # Interest Rate
        rate_range = np.linspace(0.01, 0.08, 20)
        param_values = rate_range
        param_name = "Risk-free Rate"
    
    # Calculate sensitivity (simplified - would need more complex analysis for structured products)
    if product_type == "Capital Protected Note":
        sensitivity_values = []
        for param in param_values:
            if sensitivity_param == "Volatility":
                option_val = bs_price(S0, S0, T, r, param, "call")
            elif sensitivity_param == "Time to Maturity":
                option_val = bs_price(S0, S0, param, r, sigma, "call")
            else:
                option_val = bs_price(S0, S0, T, param, sigma, "call")
                
            sensitivity_values.append(option_val)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=param_values, y=sensitivity_values, mode='lines+markers',
                                    name=f'Option Component Value', line=dict(width=3)))
        fig_sens.update_layout(title=f"Sensitivity to {param_name}",
                             xaxis_title=param_name, yaxis_title="Option Value ($)")
        st.plotly_chart(fig_sens, use_container_width=True)
    
    # Educational Summary
    with st.expander("Key Concepts Summary"):
        if product_type == "Reverse Convertible":
            st.markdown("""
            **Reverse Convertible Notes:**
            - High yield in exchange for downside equity risk
            - Economically equivalent to: Bond + Short Put Option
            - Risk: May receive depreciated stock instead of cash
            - Use case: Income generation in stable/bullish markets
            """)
        elif product_type == "Autocallable Note":
            st.markdown("""
            **Autocallable Notes:**
            - Early redemption feature if performance conditions met
            - Enhanced coupons but potential downside exposure
            - Complex path-dependent payoff structure
            - Risk: Full equity exposure if barriers breached
            """)
        elif product_type == "Capital Protected Note":
            st.markdown("""
            **Capital Protected Notes:**
            - Combines zero-coupon bond + call options
            - Guarantees principal return plus equity upside
            - Trade-off: Limited upside participation
            - Decomposition: Protection + Leveraged equity exposure
            """)
        else:
            st.markdown("""
            **Barrier Options:**
            - Path-dependent options with knock-in/knock-out features
            - Cheaper than vanilla options due to barrier risk
            - Used in structured products to reduce hedging costs
            - Risk: Discontinuous payoffs near barrier levels
            """)

# Run the page
if __name__ == "__main__":
    show_structured_products_page()
