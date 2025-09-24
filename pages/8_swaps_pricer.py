import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Swap Pricing Functions ---
def discount_factor(rate, time):
    """Calculate discount factor for given rate and time"""
    return np.exp(-rate * time)

def forward_rate(spot_rates, times):
    """Calculate forward rates from spot rates"""
    forwards = []
    for i in range(1, len(times)):
        t1, t2 = times[i-1], times[i]
        r1, r2 = spot_rates[i-1], spot_rates[i]
        forward = ((1 + r2 * t2) / (1 + r1 * t1) - 1) / (t2 - t1)
        forwards.append(forward)
    return forwards

def swap_pv_legs(notional, fixed_rate, spot_rates, times, payment_freq=0.5):
    """Calculate present value of fixed and floating legs"""
    # Payment periods
    payment_times = np.arange(payment_freq, times[-1] + payment_freq, payment_freq)
    
    # Fixed leg PV
    fixed_pv = 0
    for t in payment_times:
        if t <= times[-1]:
            # Interpolate spot rate for payment time
            rate = np.interp(t, times, spot_rates)
            df = discount_factor(rate, t)
            fixed_pv += notional * fixed_rate * payment_freq * df
    
    # Floating leg PV (using forward rates)
    floating_pv = 0
    for i, t in enumerate(payment_times):
        if t <= times[-1]:
            # Simple forward rate approximation
            if i == 0:
                forward = spot_rates[0]
            else:
                prev_t = payment_times[i-1] if i > 0 else 0
                rate_prev = np.interp(prev_t, times, spot_rates) if prev_t > 0 else spot_rates[0]
                rate_curr = np.interp(t, times, spot_rates)
                forward = rate_curr  # Simplified - in practice would use proper forward calculation
            
            df = discount_factor(np.interp(t, times, spot_rates), t)
            floating_pv += notional * forward * payment_freq * df
    
    return fixed_pv, floating_pv

def fair_swap_rate(notional, spot_rates, times, payment_freq=0.5):
    """Calculate fair swap rate (where PV of both legs are equal)"""
    payment_times = np.arange(payment_freq, times[-1] + payment_freq, payment_freq)
    
    # Annuity factor (sum of discount factors)
    annuity = 0
    for t in payment_times:
        if t <= times[-1]:
            rate = np.interp(t, times, spot_rates)
            df = discount_factor(rate, t)
            annuity += payment_freq * df
    
    # Fair swap rate
    df_final = discount_factor(spot_rates[-1], times[-1])
    fair_rate = (1 - df_final) / annuity
    
    return fair_rate

# --- Streamlit App ---
def show_swaps_page():
    st.title("💰 Interest Rate Swaps Pricer & Visualizer")

    st.markdown("""
    This tool calculates the value of an **interest rate swap (IRS)**, which exchanges fixed-rate payments for floating-rate payments.  
    It shows the **present value of each leg**, the **net value (NPV)**, and the **fair fixed rate** that makes the swap initially neutral.  
    You can also visualize the **yield curve**, **cash flow profile**, and explore **how rate changes impact swap value**.
    """)
    
    # --- Input Parameters ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Swap Parameters")
        notional = st.number_input("Notional Amount ($M)", value=100, min_value=1) * 1_000_000
        maturity = st.slider("Swap Maturity (years)", 1, 10, 5)
        payment_freq = st.selectbox("Payment Frequency", [0.25, 0.5, 1.0], index=1, 
                                   format_func=lambda x: f"{'Quarterly' if x==0.25 else 'Semi-annual' if x==0.5 else 'Annual'}")
        
    with col2:
        st.subheader("Market Conditions")
        base_rate = st.slider("Base Interest Rate (%)", 0.0, 10.0, 3.0, 0.25) / 100
        curve_steepness = st.slider("Yield Curve Steepness", -2.0, 4.0, 1.0, 0.25) / 100
        fixed_rate_input = st.number_input("Fixed Rate (% - leave 0 for fair rate)", 0.0, 15.0, 0.0, 0.25) / 100
    
    # --- Generate Yield Curve ---
    times = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    times = times[times <= maturity + 2]  # Extend slightly beyond maturity
    spot_rates = base_rate + curve_steepness * np.sqrt(times)  # Simple curve model
    
    # --- Calculate Fair Swap Rate ---
    fair_rate = fair_swap_rate(notional, spot_rates, times, payment_freq)
    
    if fixed_rate_input == 0:
        fixed_rate = fair_rate
        st.info(f"Using Fair Swap Rate: {fair_rate:.3%}")
    else:
        fixed_rate = fixed_rate_input
        st.info(f"Using Custom Fixed Rate: {fixed_rate:.3%} (Fair Rate: {fair_rate:.3%})")
    
    # --- Calculate Swap Valuation ---
    fixed_pv, floating_pv = swap_pv_legs(notional, fixed_rate, spot_rates, times, payment_freq)
    swap_npv = floating_pv - fixed_pv  # NPV for receiver of floating
    
    # --- Results Dashboard ---
    st.subheader("Swap Valuation Results")
    st.markdown("See the **PV of each leg**, the **net value of the swap**, and the **fair fixed rate**.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fixed Leg PV", f"${fixed_pv:,.0f}")
    col2.metric("Floating Leg PV", f"${floating_pv:,.0f}")
    col3.metric("Swap NPV", f"${swap_npv:,.0f}", 
               delta=f"{'Positive' if swap_npv > 0 else 'Negative' if swap_npv < 0 else 'Neutral'}")
    col4.metric("Fair Swap Rate", f"{fair_rate:.3%}")
    
    # --- Yield Curve Visualization ---
    st.subheader("📈 Yield Curve")
    st.markdown("Visualize the spot rate curve and compare **fair vs fixed rate**.")
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=times, y=spot_rates*100,
        mode='lines+markers',
        name='Spot Rate Curve',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig_curve.add_hline(y=fair_rate*100, line_dash="dash", line_color="red",
                       annotation_text=f"Fair Swap Rate: {fair_rate:.3%}")
    
    if fixed_rate_input > 0:
        fig_curve.add_hline(y=fixed_rate*100, line_dash="dash", line_color="green",
                           annotation_text=f"Fixed Rate: {fixed_rate:.3%}")
    
    fig_curve.update_layout(
        title="Interest Rate Term Structure",
        xaxis_title="Time to Maturity (years)",
        yaxis_title="Interest Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig_curve, use_container_width=True)
    
    # --- Swap Cash Flow Analysis ---
    st.subheader("💸 Cash Flow Analysis")
    
    # Generate cash flow schedule
    payment_times = np.arange(payment_freq, maturity + payment_freq, payment_freq)
    cf_data = []
    
    cumulative_fixed = 0
    cumulative_floating = 0
    
    for i, t in enumerate(payment_times):
        if t <= maturity:
            # Fixed payment
            fixed_payment = notional * fixed_rate * payment_freq
            
            # Floating payment (estimated using forward rates)
            floating_rate_est = np.interp(t, times, spot_rates)
            floating_payment = notional * floating_rate_est * payment_freq
            
            # Net payment (positive = receive, negative = pay)
            net_payment = floating_payment - fixed_payment
            
            cumulative_fixed += fixed_payment
            cumulative_floating += floating_payment
            
            cf_data.append({
                'Payment Date': f"Period {i+1} ({t:.2f}y)",
                'Fixed Payment': fixed_payment,
                'Floating Payment': floating_payment,
                'Net Cash Flow': net_payment,
                'Time': t
            })
    
    cf_df = pd.DataFrame(cf_data)
    
    # Cash flow chart
    fig_cf = go.Figure()
    
    fig_cf.add_trace(go.Bar(
        name='Fixed Payments (Pay)',
        x=cf_df['Payment Date'],
        y=-cf_df['Fixed Payment'],
        marker_color='red',
        opacity=0.7
    ))
    
    fig_cf.add_trace(go.Bar(
        name='Floating Payments (Receive)',
        x=cf_df['Payment Date'],
        y=cf_df['Floating Payment'],
        marker_color='blue',
        opacity=0.7
    ))
    
    fig_cf.add_trace(go.Scatter(
        name='Net Cash Flow',
        x=cf_df['Payment Date'],
        y=cf_df['Net Cash Flow'],
        mode='lines+markers',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    fig_cf.update_layout(
        title="Swap Cash Flow Profile (Pay Fixed, Receive Floating)",
        xaxis_title="Payment Periods",
        yaxis_title="Cash Flow ($)",
        height=500,
        barmode='relative'
    )
    
    st.plotly_chart(fig_cf, use_container_width=True)
    
    # --- Sensitivity Analysis ---
    st.subheader("📊 Sensitivity Analysis")
    
    sensitivity_type = st.selectbox("Choose Sensitivity Analysis", 
                                   ["Rate Shift (Parallel)", "Curve Steepening", "Fixed Rate Impact"])
    
    if sensitivity_type == "Rate Shift (Parallel)":
        rate_shifts = np.linspace(-200, 200, 21)  # -2% to +2% in bps
        npv_values = []
        
        for shift in rate_shifts:
            shifted_rates = spot_rates + (shift / 10000)  # Convert bps to decimal
            fixed_pv_shift, floating_pv_shift = swap_pv_legs(notional, fixed_rate, shifted_rates, times, payment_freq)
            npv_shift = floating_pv_shift - fixed_pv_shift
            npv_values.append(npv_shift)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=rate_shifts,
            y=npv_values,
            mode='lines+markers',
            name='Swap NPV',
            line=dict(color='purple', width=3)
        ))
        
        fig_sens.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Current Rates")
        fig_sens.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig_sens.update_layout(
            title="Swap NPV Sensitivity to Parallel Rate Shifts",
            xaxis_title="Rate Shift (basis points)",
            yaxis_title="Swap NPV ($)",
            height=400
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
        # Calculate DV01 (Dollar Value of 1 basis point)
        if len(npv_values) >= 3:
            dv01 = (npv_values[11] - npv_values[9]) / 2  # Central difference for 1bp
            st.metric("DV01 (1bp impact)", f"${dv01:,.0f}")
    
    elif sensitivity_type == "Curve Steepening":
        steep_values = np.linspace(-3, 3, 13)  # Different steepness levels
        npv_values = []
        
        for steep in steep_values:
            steep_rates = base_rate + (steep / 100) * np.sqrt(times)
            fixed_pv_steep, floating_pv_steep = swap_pv_legs(notional, fixed_rate, steep_rates, times, payment_freq)
            npv_steep = floating_pv_steep - fixed_pv_steep
            npv_values.append(npv_steep)
        
        fig_steep = go.Figure()
        fig_steep.add_trace(go.Scatter(
            x=steep_values,
            y=npv_values,
            mode='lines+markers',
            name='Swap NPV',
            line=dict(color='orange', width=3)
        ))
        
        fig_steep.add_vline(x=curve_steepness*100, line_dash="dash", line_color="red", annotation_text="Current Steepness")
        fig_steep.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig_steep.update_layout(
            title="Swap NPV Sensitivity to Curve Steepness",
            xaxis_title="Curve Steepness (%)",
            yaxis_title="Swap NPV ($)",
            height=400
        )
        
        st.plotly_chart(fig_steep, use_container_width=True)
    
    # --- Educational Summary ---
    with st.expander("📚 Key Concepts Summary"):
        st.markdown("""
        **Interest Rate Swap Basics:**
        - **Fixed Leg**: Pays a predetermined rate throughout the swap's life
        - **Floating Leg**: Pays a variable rate that resets periodically
        - **Fair Swap Rate**: The fixed rate that makes both legs equal in value at inception
        - **NPV**: Net Present Value - difference between floating and fixed leg values
        
        **Risk Factors:**
        - **Interest Rate Risk**: Swap value changes with rate movements
        - **Curve Risk**: Sensitivity to yield curve shape changes  
        - **DV01**: Dollar impact of a 1 basis point rate change
        
        **Use Cases:**
        - **Hedging**: Convert floating rate debt to fixed (or vice versa)
        - **Speculation**: Take views on interest rate direction
        - **Asset-Liability Matching**: Align cash flow characteristics
        """)

# Run the page
if __name__ == "__main__":
    show_swaps_page()
