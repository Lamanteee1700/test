import streamlit as st

st.title("📖 Black-Scholes Model: Theory & Assumptions")

st.markdown("""
### Assumptions
- The stock follows a **geometric Brownian motion** with constant drift and volatility.  
- No arbitrage opportunities.  
- Markets are frictionless (no transaction costs, infinite divisibility).  
- Constant risk-free interest rate.  
- European-style options (exercise only at expiry).  

### Black–Scholes PDE
The option price \\( V(S,t) \\) satisfies:
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} 
+ rS \frac{\partial V}{\partial S} - rV = 0
$$

### Closed-form Solution
For a **European Call**:
$$
C(S, t) = S N(d_1) - K e^{-rT} N(d_2)
$$

For a **European Put**:
$$
P(S, t) = K e^{-rT} N(-d_2) - S N(-d_1)
$$

where:
$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

### Greeks
- **Delta (Δ)**: sensitivity to stock price.  
- **Gamma (Γ)**: rate of change of Delta.  
- **Vega (ν)**: sensitivity to volatility.  
- **Theta (Θ)**: sensitivity to time.  
- **Rho (ρ)**: sensitivity to interest rate.

### Strike Price Scenarios  
- **ATM (At the Money)** → strike ≈ current spot price.  
- **Deep ITM (In the Money)**  
 - Call: strike ≈ 70% of spot (much cheaper than stock).  
 - Put: strike ≈ 130% of spot (much higher than stock).  
- **Deep OTM (Out of the Money)**  
 - Call: strike ≈ 130% of spot (much more expensive than stock).  
 - Put: strike ≈ 70% of spot (much cheaper than stock).  
""")
