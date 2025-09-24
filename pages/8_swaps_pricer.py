import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Swap Pricing Functions ---
def discount_factor(rate, time):
    """Calculate discount factor for a given rate and time.
    DF represents the present value of $1 received at time t discounted at rate r."""
    return np.exp(-rate * time)

def forward_rate(spot_rates, times):
    """Calculate forward rates from spot rates using:
       f(t1,t2) = (DF(t1)/DF(t2) - 1) / (t2 - t1)
       Gives expected short rate between t1 and t2 implied by the curve."""
    dfs = discount_factor(np.array(spot_rates), np.array(times))
    forwards = [spot_rates[0]]  # First period forward approximated as spot
    for i in range(1, len(times)):
        fwd = (dfs[i-1]/dfs[i] - 1) / (times[i] - times[i-1])
        forwards.append(fwd)
    return forwards

def swap_pv_legs(notional, fixed_rate, spot_rates, times, payment_freq=0.5):
    """
    Calculate PV of fixed and floating legs.
    Fixed leg: discounted sum of fixed payments.
    Floating leg: discounted sum of expected floating payments using forward rates.
    """
    payment_times = np.arange(payment_freq, times[-1] + payment_freq, payment_freq)
    
    # --- Fixed Leg ---
    fixed_pv = 0
    for t in payment_times:
        if t <= times[-1]:
            rate = np.interp(t, times, spot_rates)
            df = discount_factor(rate, t)
            fixed_pv += notional * fixed_rate * payment_freq * df
    
    # --- Floating Leg using proper forward rates ---
    fwd_rates = forward_rate(spot_rates, times)
    floating_pv = 0
    for i, t in enumerate(payment_times):
        if t <= times[-1]:
            fwd = fwd_rates[i]
            df = discount_factor(np.interp(t, times, spot_rates), t)
            floating_pv += notional * fwd * payment_freq * df
    
    return fixed_pv, floating_pv

def fair_swap_rate(notional, spot_rates, times, payment_freq=0.5):
    """
    Calculate fair swap rate where PV of fixed leg = PV of floating leg.
    Conceptually: fixed rate = (1 - PV of final DF) / sum of discounted periods (annuity)
    """
    payment_times = np.arange(payment_freq, times[-1] + payment_freq, payment_freq)
    annuity = 0
    for t in payment_times:
        if t <= times[-1]:
            rate = np.interp(t, times, spot_rates)
            df = discount_factor(rate, t)
            annuity += payment_freq * df
    
    df_final = discount_factor(spot_rates[-1], times[-1])
    fair_rate = (1 - df_final) / annuity
    return fair_rate

# --- Streamlit App ---
def show_swaps_page():
    st.title("💰 Interest Rate Swaps Pricer & Visualizer")
    
    st.markdown("""
    **Interest Rate Swaps (IRS)** allow exchanging fixed-rate payments for floating-rate payments.
    This tool calculates swap valuation, fair rate, cash flows, and sensitivity, with intuitive visualizations.
    """)

    # --- Input Parameters ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Swap Parameters")
        st.markdown("Define the main characteristics of your swap. This affects timing and size of payments.")
        notional = st.number_input("Notional Amount ($M)", value=100, min_value=1) * 1_000_000
        maturity = st.slider("Swap Maturity (years)", 1, 10, 5)
        payment_freq = st.selectbox(
            "Payment Frequency", [0.25, 0.5, 1.0], index=1, 
            format_func=lambda x: f"{'Quarterly' if x==0.25 else 'Semi-annual' if x==0.5 else 'Annual'}"
        )
    
    with col2:
        st.subheader("Market Conditions")
        st.markdown("Set the base rate and curve shape. Optionally input a fixed rate, otherwise the fair rate is used.")
        base_rate = st.slider("Base Interest Rate (%)", 0.0, 10.0, 3.0, 0.25) / 100
        curve_steepness = st.slider("Yield Curve Steepness", -2.0, 4.0, 1.0, 0.25) / 100
        fixed_rate_input = st.number_input("Fixed Rate (% - leave 0 for fair rate)", 0.0, 15.0, 0.0, 0.25) / 100

    # --- Generate Yield Curve ---
    times = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    times = times[times <= maturity + 2]  # Slightly beyond maturity
    spot_rates = base_rate + curve_steepness * np.sqrt(times)  # Simple curve model
    st.markdown("📊 Yield curve generated from base rate and curve steepness.")

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
    swap_npv = floating_pv - fixed_pv
    st.subheader("Swap Valuation Results")
    st.markdown("PV of each leg and swap NPV calculated. Positive NPV indicates value for floating receiver.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fixed Leg PV", f"${fixed_pv:,.0f}")
    col2.metric("Floating Leg PV", f"${floating_pv:,.0f}")
    col3.metric("Swap NPV", f"${swap_npv:,.0f}", delta=f"{'Positive' if swap_npv > 0 else 'Negative' if swap_npv < 0 else 'Neutral'}")
    col4.metric("Fair Swap Rate", f"{fair_rate:.3%}")

    # --- Yield Curve Visualization ---
    st.subheader("📈 Yield Curve")
    st.markdown("Spot rates vs maturity. Red = fair swap rate, green = chosen fixed rate (if any).")
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=times, y=spot_rates*100, mode='lines+markers', name='Spot Rate Curve',
        line=dict(color='blue', width=3), marker=dict(size=8)
    ))
    fig_curve.add_hline(y=fair_rate*100, line_dash="dash", line_color="red", annotation_text=f"Fair Swap Rate: {fair_rate:.3%}")
    if fixed_rate_input > 0:
        fig_curve.add_hline(y=fixed_rate*100, line_dash="dash", line_color="green", annotation_text=f"Fixed Rate: {fixed_rate:.3%}")
    fig_curve.update_layout(title="Interest Rate Term Structure", xaxis_title="Time to Maturity (years)", yaxis_title="Interest Rate (%)", height=400)
    st.plotly_chart(fig_curve, use_container_width=True)

    # --- Swap Cash Flow Analysis ---
    st.subheader("💸 Cash Flow Analysis")
    st.markdown("Fixed vs floating payments per period. Net cash flow = floating received - fixed paid.")

    payment_times = np.arange(payment_freq, maturity + payment_freq, payment_freq)
    cf_data = []
    cumulative_fixed = 0
    cumulative_floating = 0
    fwd_rates = forward_rate(spot_rates, times)

    for i, t in enumerate(payment_times):
        if t <= maturity:
            fixed_payment = notional * fixed_rate * payment_freq
            floating_payment = notional * fwd_rates[i] * payment_freq
            net_payment = floating_payment - fixed_payment
            cumulative_fixed += fixed_payment
            cumulative_floating += floating_payment
            cf_data.append({'Payment Date': f"Period {i+1} ({t:.2f}y)", 'Fixed Payment': fixed_payment,
                            'Floating Payment': floating_payment, 'Net Cash Flow': net_payment, 'Time': t})
    
    cf_df = pd.DataFrame(cf_data)
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(name='Fixed Payments (Pay)', x=cf_df['Payment Date'], y=-cf_df['Fixed Payment'], marker_color='red', opacity=0.7))
    fig_cf.add_trace(go.Bar(name='Floating Payments (Receive)', x=cf_df['Payment Date'], y=cf_df['Floating Payment'], marker_color='blue', opacity=0.7))
    fig_cf.add_trace(go.Scatter(name='Net Cash Flow', x=cf_df['Payment Date'], y=cf_df['Net Cash Flow'], mode='lines+markers', line=dict(color='green', width=3), marker=dict(size=8)))
    fig_cf.update_layout(title="Swap Cash Flow Profile (Pay Fixed, Receive Floating)", xaxis_title="Payment Periods", yaxis_title="Cash Flow ($)", height=500, barmode='relative')
    st.plotly_chart(fig_cf, use_container_width=True)

    # --- Sensitivity Analysis ---
    st.subheader("📊 Sensitivity Analysis")
    st.markdown("Analyze how swap NPV responds to interest rate changes or curve steepness.")

    sensitivity_type = st.selectbox("Choose Sensitivity Analysis", ["Rate Shift (Parallel)", "Curve Steepening", "Fixed Rate Impact"])

    if sensitivity_type == "Rate Shift (Parallel)":
        rate_shifts = np.linspace(-200, 200, 21)  # -2% to +2% in bps
        npv_values = []
        for shift in rate_shifts:
            shifted_rates = spot_rates + (shift / 10000)
            fixed_pv_shift, floating_pv_shift = swap_pv_legs(notional, fixed_rate, shifted_rates, times, payment_freq)
            npv_values.append(floating_pv_shift - fixed_pv_shift)
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=rate_shifts, y=npv_values, mode='lines+markers', name='Swap NPV', line=dict(color='purple', width=3)))
        fig_sens.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Current Rates")
        fig_sens.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sens.update_layout(title="Swap NPV Sensitivity to Parallel Rate Shifts", xaxis_title="Rate Shift (bps)", yaxis_title="Swap NPV ($)", height=400)
        st.plotly_chart(fig_sens, use_container_width=True)

        # DV01
        if len(npv_values) >= 3:
            dv01 = (npv_values[11] - npv_values[9]) / 2  # Central difference approx 1bp
            st.metric("DV01 (1bp impact)", f"${dv01:,.0f}")

    elif sensitivity_type == "Curve Steepening":
        steep_values = np.linspace(-3, 3, 13)
        npv_values = []
        for steep in steep_values:
            steep_rates = base_rate + (steep / 100) * np.sqrt(times)
            fixed_pv_steep, floating_pv_steep = swap_pv_legs(notional, fixed_rate, steep_rates, times, payment_freq)
            npv_values.append(floating_pv_steep - fixed_pv_steep)
        fig_steep = go.Figure()
        fig_steep.add_trace(go.Scatter(x=steep_values, y=npv_values, mode='lines+markers', name='Swap NPV', line=dict(color='orange', width=3)))
        fig_steep.add_vline(x=curve_steepness*100, line_dash="dash", line_color="red", annotation_text="Current Steepness")
        fig_steep.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_steep.update_layout(title="Swap NPV Sensitivity to Curve Steepness", xaxis_title="Curve Steepness (%)", yaxis_title="Swap NPV ($)", height=400)
        st.plotly_chart(fig_steep, use_container_width=True)

    # --- Educational Summary ---
    with st.expander("📚 Key Concepts Summary"):
        st.markdown("""
        **Interest Rate Swap Basics:**
        - **Fixed Leg**: Pays a predetermined rate throughout the swap's life
        - **Floating Leg**: Pays a variable rate that resets periodically, approximated by forward rates
        - **Fair Swap Rate**: The fixed rate making both legs equal in value at inception
        - **NPV**: Net Present Value = floating PV - fixed PV
        
        **Risk Factors:**
        - **Interest Rate Risk**: Swap value changes with rate movements
        - **Curve Risk**: Sensitivity to yield curve shape changes  
        - **DV01**: Dollar impact of a 1 basis point rate change
        
        **Use Cases:**
        - **Hedging**: Convert floating rate debt to fixed or vice versa
        - **Speculation**: Take views on interest rate direction
        - **Asset-Liability Matching**: Align cash flow characteristics
        """)

# Run app
if __name__ == "__main__":
    show_swaps_page()
