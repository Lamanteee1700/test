import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from utils import bs_price, greeks, d1, d2

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Derivatives Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Navigation Sidebar ---
st.sidebar.title("📊 Financial Derivatives Dashboard")
page = st.sidebar.radio(
    "Select Analysis Tool:",
    ["🎯 Options Pricer", "💰 Swaps Pricer", "🏗️ Structured Products", "📚 About"]
)

try:
    price = bs_price(S, K, T, r, sigma, option=option_type)
    delta, gamma, vega, theta, rho = greeks(S, K, T, r, sigma, option=option_type)
except Exception as e:
    st.error(f"❌ Calculation error: {str(e)}")
    return

# Results Display
st.subheader("💡 Option Valuation Results")

result_col1, result_col2, result_col3, result_col4 = st.columns(4)
result_col1.metric(f"{option_type.capitalize()} Price", f"${price:.3f}")
result_col2.metric("Intrinsic Value", f"${max(0, (S-K) if option_type=='call' else (K-S)):.3f}")
result_col3.metric("Time Value", f"${max(0, price - max(0, (S-K) if option_type=='call' else (K-S))):.3f}")
result_col4.metric("Moneyness", f"{S/K:.3f}")

# Greeks Dashboard
st.subheader("📊 Greeks Dashboard")

greeks_col1, greeks_col2 = st.columns(2)

with greeks_col1:
    st.metric("Delta (Δ)", f"{delta:.4f}", help="Price sensitivity to underlying asset")
    st.metric("Gamma (Γ)", f"{gamma:.4f}", help="Delta sensitivity to underlying asset") 
    st.metric("Vega (ν)", f"{vega:.4f}", help="Price sensitivity to volatility")

with greeks_col2:
    st.metric("Theta (Θ)", f"{theta:.4f}", help="Price decay per day")
    st.metric("Rho (ρ)", f"{rho:.4f}", help="Price sensitivity to interest rates")

# Greeks Visualization
st.subheader("📈 Greeks Sensitivity Analysis")

# Greek selection
greek_choices = st.multiselect(
    "Choose Greeks to plot:",
    ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    default=["Delta", "Gamma"]
)

if greek_choices:
    # Price range for analysis
    price_range = st.slider("Price Range for Analysis", 0.5, 2.0, (0.8, 1.2), 0.1)
    S_min, S_max = S * price_range[0], S * price_range[1]
    S_range = np.linspace(S_min, S_max, 100)
    
    # Calculate Greeks across price range
    greeks_dict = {g: [] for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}

    for S_val in S_range:
        try:
            d, g, v, t, r_ = greeks(S_val, K, T, r, sigma, option=option_type)
            greeks_dict["Delta"].append(d)
            greeks_dict["Gamma"].append(g)
            greeks_dict["Vega"].append(v)
            greeks_dict["Theta"].append(t)
            greeks_dict["Rho"].append(r_)
        except:
            # Handle edge cases
            greeks_dict["Delta"].append(np.nan)
            greeks_dict["Gamma"].append(np.nan)
            greeks_dict["Vega"].append(np.nan)
            greeks_dict["Theta"].append(np.nan)
            greeks_dict["Rho"].append(np.nan)

    # Plot Greeks
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, g in enumerate(greek_choices):
        ax.plot(S_range, greeks_dict[g], label=g, color=colors[i % len(colors)], linewidth=2)
    
    ax.axvline(S, color="black", linestyle="--", alpha=0.7, label="Current Price")
    ax.axvline(K, color="red", linestyle=":", alpha=0.7, label="Strike Price")
    ax.set_xlabel("Stock Price ($)")
    ax.set_ylabel("Greek Value")
    ax.set_title(f"Greeks Sensitivity - {option_type.capitalize()} Option")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Implied Volatility Calculator
st.subheader("📈 Implied Volatility Calculator")

iv_col1, iv_col2 = st.columns(2)

with iv_col1:
    market_price = st.number_input("Enter Market Option Price ($)", min_value=0.0, step=0.01)

with iv_col2:
    if market_price > 0:
        def option_price_given_vol(vol):
            try:
                return bs_price(S, K, T, r, vol, option_type) - market_price
            except:
                return float('inf')
        
        try:
            implied_vol = brentq(option_price_given_vol, 0.001, 5.0)
            st.success(f"**Implied Volatility: {implied_vol:.2%}**")
            
            # Compare with input volatility
            vol_diff = implied_vol - sigma
            if abs(vol_diff) > 0.05:  # 5% difference
                if vol_diff > 0:
                    st.warning(f"⚠️ IV is {vol_diff:.2%} higher than assumed volatility")
                else:
                    st.info(f"ℹ️ IV is {abs(vol_diff):.2%} lower than assumed volatility")
                    
        except ValueError:
            st.error("❌ Could not calculate implied volatility with given inputs.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# === SWAPS PRICER PAGE ===
def show_swaps_page():
# Import and execute the swaps functionality
try:
    from swaps_pricer import show_swaps_page
    show_swaps_page()
except ImportError:
    st.error("❌ swaps_pricer.py module not found. Please ensure it's in the same directory.")
    st.markdown("""
    **To use the Swaps Pricer:**
    1. Save the swaps pricing code as `swaps_pricer.py` in the same directory
    2. The swaps page will load automatically
    
    **Or copy the swaps functionality directly into this file.**
    """)

# === STRUCTURED PRODUCTS PAGE ===  
def show_structured_products_page():
try:
    from structured_products import show_structured_products_page
    show_structured_products_page()
except ImportError:
    st.error("❌ structured_products.py module not found. Please ensure it's in the same directory.")
    st.markdown("""
    **To use the Structured Products Builder:**
    1. Save the structured products code as `structured_products.py` in the same directory
    2. The structured products page will load automatically
    """)
    
    # Fallback basic structured products demo
    st.subheader("Basic Structured Products Demo")
    st.write("This would show the structured products functionality once the module is available.")
    
    # Simple reverse convertible example
    S = st.slider("Asset Price", 50.0, 150.0, 100.0)
    barrier = st.slider("Barrier Level", 50.0, 95.0, 70.0)
    coupon = st.slider("Coupon Rate (%)", 5.0, 15.0, 8.0) / 100
    
    if S >= barrier:
        payoff = 100 + coupon * 100
        st.success(f"Payoff: ${payoff:.2f} (Principal + Coupon)")
    else:
        shares = 100 / 100  # Simplified
        payoff = shares * S + coupon * 100
        st.warning(f"Payoff: ${payoff:.2f} (Stock Value + Coupon)")
        
    st.line_chart(pd.DataFrame({
        'Asset Price': range(60, 140),
        'Payoff': [shares * price + coupon * 100 if price < barrier else 100 + coupon * 100 
                  for price in range(60, 140)]
    }).set_index('Asset Price'))

# === ABOUT PAGE ===
def show_about_page():
st.title("📚 About Financial Derivatives Dashboard")

st.markdown("""
This dashboard provides interactive tools for pricing and analyzing financial derivatives,
with a focus on **pedagogical clarity** and **practical application**.

## 🎯 Options Pricer
- **Black-Scholes pricing** for European calls and puts
- **Greeks analysis** with interactive visualizations  
- **Implied volatility** calculation from market prices
- **Real-time data** integration via Yahoo Finance

## 💰 Swaps Pricer  
- **Interest Rate Swap** valuation and fair rate calculation
- **Yield curve** analysis and sensitivity testing
- **Cash flow** visualization and NPV calculation
- **Risk metrics** including DV01 and curve risk

## 🏗️ Structured Products Builder
- **Reverse Convertibles** with barrier analysis and risk metrics
- **Autocallable Notes** with Monte Carlo simulation
- **Capital Protected Notes** showing bond/option decomposition
- **Barrier Options** with path-dependent pricing
- **Payoff visualization** and component breakdown

## 🔗 Key Features
- **Interactive parameters** with real-time updates
- **Educational explanations** for complex concepts  
- **Professional visualizations** using Plotly and Matplotlib
- **Risk analysis** tools for sensitivity and scenario testing

## 📖 Educational Focus
This tool is designed to help users understand:
- How derivative prices respond to market changes
- The relationship between risk factors and valuation
- Real-world applications in risk management and trading
- The mathematical foundations behind pricing models

## ⚠️ Disclaimer
This tool is for **educational purposes only**. All calculations are based on simplified models
and should not be used for actual trading or investment decisions without proper validation
and risk management procedures.

---
**Built with:** Streamlit, NumPy, SciPy, Plotly, YFinance  
**Models:** Black-Scholes, Present Value Discounting, Monte Carlo Methods
""")

# Technical Details Expander
with st.expander("🔧 Technical Implementation Details"):
    st.markdown("""
    **Options Pricing:**
    - Black-Scholes-Merton model for European options
    - Greeks calculated via analytical formulas
    - Implied volatility solved using Brent's method
    
    **Swaps Pricing:**  
    - Present value approach with term structure modeling
    - Fair swap rate calculation via annuity factors
    - Sensitivity analysis through finite difference methods
    
    **Data Sources:**
    - Yahoo Finance API for real-time equity prices
    - User-defined parameters for interest rates and volatility
    - Simulated yield curves based on standard models
    """)

# === MAIN APP LOGIC ===
def main():
# Show selected page
if page == "🎯 Options Pricer":
    show_options_page()
elif page == "💰 Swaps Pricer":
    show_swaps_page()  
elif page == "🏗️ Structured Products":
    show_structured_products_page()
elif page == "📚 About":
    show_about_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Financial Derivatives Dashboard**")
st.sidebar.markdown("*Educational tool for derivatives analysis*")

if __name__ == "__main__":
main()
