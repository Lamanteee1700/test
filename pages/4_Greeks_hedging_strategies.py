# pages/3_Greeks_Hedging_Strategy.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from utils import bs_price, greeks
from datetime import date

st.set_page_config(page_title="Greeks Hedging Strategy", layout="wide")
st.title("🧮 Multi-Asset Greeks Hedging Strategy")

st.markdown("""
This interactive page helps you **build or load a multi-asset portfolio** (stocks + options), 
inspect aggregated Greeks, and experiment with **Delta** and **Gamma hedging** steps.  
Inspired by the practical workflow described in *Mastering the Greeks* (section 7).
""")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fetch_spot(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

def days_to_T(days):
    return max(days, 0) / 365.0

def option_payoff_vector(S_range, strike, option_type, qty_contracts):
    mult = qty_contracts * 100.0
    if option_type == "call":
        return np.maximum(S_range - strike, 0.0) * mult
    else:
        return np.maximum(strike - S_range, 0.0) * mult

# -----------------------------------------------------------------------------
# Portfolio mode: custom vs presets
# -----------------------------------------------------------------------------
st.subheader("Portfolio setup")

portfolio_mode = st.radio(
    "Choose portfolio mode:",
    [
        "Custom (build manually)",
        "Preset: Conservative Hedged Equity",
        "Preset: Option Seller (Income)",
        "Preset: Speculative Leveraged Options"
    ],
    index=0
)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if portfolio_mode.startswith("Preset"):
    # Load presets
    if portfolio_mode == "Preset: Conservative Hedged Equity":
        ticker = "AAPL"
        spot = fetch_spot(ticker)
        st.session_state.portfolio = [
            {"type": "stock", "ticker": ticker, "qty": 100, "spot": spot},
            {"type": "option", "underlying": ticker, "option_type": "put",
             "strike": round(spot*0.95,2), "days": 60, "qty": 1, "vol": 0.25, "spot": spot}
        ]
        st.info("Preset loaded: Long 100 AAPL shares + 1 protective put (≈95% strike, 60 days).")

    elif portfolio_mode == "Preset: Option Seller (Income)":
        ticker = "MSFT"
        spot = fetch_spot(ticker)
        st.session_state.portfolio = [
            {"type": "stock", "ticker": ticker, "qty": 100, "spot": spot},
            {"type": "option", "underlying": ticker, "option_type": "call",
             "strike": round(spot*1.05,2), "days": 30, "qty": -1, "vol": 0.22, "spot": spot}
        ]
        st.info("Preset loaded: Covered call (long 100 MSFT shares + short 1 OTM call).")

    elif portfolio_mode == "Preset: Speculative Leveraged Options":
        ticker = "TSLA"
        spot = fetch_spot(ticker)
        st.session_state.portfolio = [
            {"type": "option", "underlying": ticker, "option_type": "call",
             "strike": round(spot*1.00,2), "days": 45, "qty": 2, "vol": 0.40, "spot": spot},
            {"type": "option", "underlying": ticker, "option_type": "call",
             "strike": round(spot*1.20,2), "days": 45, "qty": -2, "vol": 0.40, "spot": spot}
        ]
        st.info("Preset loaded: Call spread on TSLA (long 2 ATM calls, short 2 OTM calls).")

else:
    # Custom mode
    st.markdown("Build your own portfolio by adding instruments from the **sidebar**:")

    st.sidebar.header("Add instrument to portfolio")

    inst_type = st.sidebar.selectbox("Instrument type", ["Stock", "Option"])

    if inst_type == "Stock":
        st.sidebar.subheader("Add stock")
        stock_ticker = st.sidebar.text_input("Ticker (stock)", value="AAPL", key="stk_ticker")
        stock_qty = st.sidebar.number_input("Quantity (shares)", value=0, step=1, key="stk_qty")
        if st.sidebar.button("Add stock"):
            spot = fetch_spot(stock_ticker)
            st.session_state.portfolio.append({
                "type": "stock",
                "ticker": stock_ticker.upper(),
                "qty": int(stock_qty),
                "spot": spot
            })
            st.sidebar.success(f"Added {stock_qty} shares of {stock_ticker.upper()} (spot {spot})")

    else:
        st.sidebar.subheader("Add option")
        opt_under = st.sidebar.text_input("Underlying ticker", value="AAPL", key="opt_under")
        opt_type = st.sidebar.selectbox("Call / Put", ["call", "put"], key="opt_type")
        strike_mode = st.sidebar.selectbox("Strike mode", ["Custom", "ATM", "Deep ITM", "Deep OTM"], key="opt_strikemode")
        opt_days = st.sidebar.number_input("Days to expiry", min_value=1, value=30, step=1, key="opt_days")
        opt_qty = st.sidebar.number_input("Quantity (contracts)", min_value=-10, value=1, step=1, key="opt_qty")
        opt_vol = st.sidebar.number_input("Implied vol (annual)", min_value=0.01, value=0.20, step=0.01, key="opt_vol")

        if st.sidebar.button("Add option"):
            spot = fetch_spot(opt_under)
            if strike_mode == "ATM":
                strike = round(spot, 2) if spot else 0.0
            elif strike_mode == "Deep ITM":
                strike = round(spot * (0.7 if opt_type == "call" else 1.3), 2) if spot else 0.0
            elif strike_mode == "Deep OTM":
                strike = round(spot * (1.3 if opt_type == "call" else 0.7), 2) if spot else 0.0
            else:
                strike = st.sidebar.number_input("Custom strike", value=round(spot or 100.0, 2), key="opt_custom_strike")
            st.session_state.portfolio.append({
                "type": "option",
                "underlying": opt_under.upper(),
                "option_type": opt_type,
                "strike": float(strike),
                "days": int(opt_days),
                "qty": int(opt_qty),
                "vol": float(opt_vol),
                "spot": spot
            })
            st.sidebar.success(f"Added {opt_qty} {opt_type.upper()} @ {strike} on {opt_under.upper()} (spot {spot})")

    if st.sidebar.button("Clear portfolio"):
        st.session_state.portfolio = []
        st.sidebar.info("Portfolio cleared")

# -----------------------------------------------------------------------------
# Portfolio display
# -----------------------------------------------------------------------------
st.subheader("Portfolio composition")
if not st.session_state.portfolio:
    st.info("Portfolio is empty — add stocks or options from the sidebar or select a preset.")
else:
    df_rows = []
    for i, inst in enumerate(st.session_state.portfolio):
        if inst["type"] == "stock":
            df_rows.append({
                "id": i,
                "type": "stock",
                "ticker": inst["ticker"],
                "qty": inst["qty"],
                "spot": inst.get("spot", None)
            })
        else:
            df_rows.append({
                "id": i,
                "type": "option",
                "underlying": inst["underlying"],
                "opt_type": inst["option_type"],
                "strike": inst["strike"],
                "days": inst["days"],
                "qty_contracts": inst["qty"],
                "vol": inst["vol"],
                "spot": inst.get("spot", None)
            })
    st.table(pd.DataFrame(df_rows).astype(str))

# -----------------------------------------------------------------------------
# Pedagogical explanations for presets
# -----------------------------------------------------------------------------
if portfolio_mode == "Preset: Conservative Hedged Equity":
    st.markdown("""
    ### 📘 Preset Explanation: Conservative Hedged Equity
    - **Composition**: Long 100 shares of AAPL + 1 protective put (≈95% of spot, 60 days).  
    - **Objective**: Participate in equity upside while limiting downside risk.  
    - **Investor Attitude**: Risk-averse investor, willing to pay a premium (put) for insurance.  
    - **Greek Profile**:  
      - Delta ≈ +100 (stock exposure).  
      - Vega > 0 (benefits from vol rise).  
      - Gamma > 0 (convexity from put).  
    - **Pedagogical Note**: Classic **hedged equity** strategy, widely used by funds.
    """)

elif portfolio_mode == "Preset: Option Seller (Income)":
    st.markdown("""
    ### 📘 Preset Explanation: Option Seller (Income)
    - **Composition**: Long 100 MSFT shares + short 1 OTM call (≈105% strike, 30 days).  
    - **Objective**: Generate income (option premium) on top of stock returns.  
    - **Investor Attitude**: Yield-seeking investor, moderately bullish but happy to sell upside.  
    - **Greek Profile**:  
      - Delta ≈ +100 minus call delta.  
      - Theta > 0 (profits from time decay).  
      - Gamma < 0 (risk if stock rallies hard).  
    - **Pedagogical Note**: Known as a **covered call**, a popular yield-enhancement tool.
    """)

elif portfolio_mode == "Preset: Speculative Leveraged Options":
    st.markdown("""
    ### 📘 Preset Explanation: Speculative Leveraged Options
    - **Composition**: Long 2 ATM calls + short 2 OTM calls on TSLA (45 days).  
    - **Objective**: Leveraged bet on moderate upside in TSLA at lower cost.  
    - **Investor Attitude**: Speculator, willing to risk premium for leveraged upside.  
    - **Greek Profile**:  
      - Delta > 0 (bullish).  
      - Gamma > 0 near ATM but capped above short strike.  
      - Vega > 0 (benefits from vol).  
    - **Pedagogical Note**: Example of a **bull call spread** — cheaper than outright calls but with capped gains.
    """)
# -----------------------------------------------------------------------------
# Global market parameters
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Global market parameters")
r = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.005)
base_range_pct = st.sidebar.slider("Price sweep around spot (%)", 10, 100, 30)

# -----------------------------------------------------------------------------
# Compute aggregated Greeks & hedging
# -----------------------------------------------------------------------------
if st.session_state.portfolio:
    st.subheader("Aggregated Greeks & Hedging Suggestions")

    total_delta, total_gamma, total_vega, total_theta, total_rho = 0, 0, 0, 0, 0
    under_spots = {}

    instruments_info = []
    for inst in st.session_state.portfolio:
        if inst["type"] == "stock":
            S_i = inst.get("spot", fetch_spot(inst["ticker"]))
            qty = inst["qty"]
            d, g, v, t_, r_ = qty, 0, 0, 0, 0
            instruments_info.append({
                "label": f"Stock {inst['ticker']}",
                "spot": S_i,
                "delta": d, "gamma": g, "vega": v, "theta": t_, "rho": r_
            })
        else:
            S_i = inst.get("spot", fetch_spot(inst["underlying"]))
            K = float(inst["strike"])
            T = days_to_T(inst["days"])
            sigma = float(inst["vol"])
            qty_contracts = int(inst["qty"])
            d, g, v, t_, r_ = greeks(S_i, K, T, r, sigma, inst["option_type"])
            scale = qty_contracts * 100
            d, g, v, t_, r_ = d*scale, g*scale, v*scale, t_*scale, r_*scale
            instruments_info.append({
                "label": f"{inst['option_type'].upper()} {inst['underlying']} K={K}",
                "spot": S_i,
                "delta": d, "gamma": g, "vega": v, "theta": t_, "rho": r_
            })

        total_delta += instruments_info[-1]["delta"]
        total_gamma += instruments_info[-1]["gamma"]
        total_vega += instruments_info[-1]["vega"]
        total_theta += instruments_info[-1]["theta"]
        total_rho += instruments_info[-1]["rho"]

        if inst["type"] == "stock":
            under_spots[inst["ticker"]] = S_i
        else:
            under_spots[inst["underlying"]] = S_i

    # Totals
    st.table(pd.DataFrame([{
        "Delta": total_delta,
        "Gamma": total_gamma,
        "Vega": total_vega,
        "Theta": total_theta,
        "Rho": total_rho
    }]).T.rename(columns={0:"Value"}).style.format("{:.3f}"))

    # Delta hedge
    st.markdown("### 1) Delta Hedge")
    hedge_shares = -total_delta
    st.info(f"To neutralize delta today, trade **{hedge_shares:.2f} shares** of the underlying (negative = sell).")

    # Gamma hedge
    st.markdown("### 2) Gamma Hedge")
    candidate_under = st.selectbox("Candidate option underlying", options=list(under_spots.keys()))
    cand_spot = under_spots[candidate_under]
    cand_strike_mode = st.selectbox("Candidate strike", ["ATM", "OTM (1.05x)", "ITM (0.95x)", "Custom"])
    if cand_strike_mode == "ATM":
        cand_strike = round(cand_spot, 2)
    elif cand_strike_mode == "OTM (1.05x)":
        cand_strike = round(cand_spot*1.05, 2)
    elif cand_strike_mode == "ITM (0.95x)":
        cand_strike = round(cand_spot*0.95, 2)
    else:
        cand_strike = st.number_input("Custom strike", value=round(cand_spot, 2))
    cand_days = st.number_input("Candidate days to expiry", min_value=1, value=30)
    cand_vol = st.number_input("Candidate vol", min_value=0.01, value=0.2)
    cand_type = st.selectbox("Candidate type", ["call", "put"])

    T_c = days_to_T(cand_days)
    _, g_c, *_ = greeks(cand_spot, cand_strike, T_c, r, cand_vol, cand_type)
    gamma_per_contract = g_c * 100
    if gamma_per_contract != 0:
        qty_needed = -total_gamma / gamma_per_contract
        st.success(f"Approx. {qty_needed:.2f} contracts needed to neutralize gamma.")
    else:
        st.warning("Candidate has near-zero gamma — pick another strike/expiry.")

    # -----------------------------------------------------------------------------
    # Payoff visualization
    # -----------------------------------------------------------------------------
    st.subheader("Payoff Visualization (at expiry)")
    main_under = list(under_spots.keys())[0]
    main_spot = under_spots[main_under]
    S_plot = np.linspace(main_spot*(1-base_range_pct/100), main_spot*(1+base_range_pct/100), 200)

    total_payoff = np.zeros_like(S_plot)
    for inst in st.session_state.portfolio:
        if inst["type"] == "stock":
            S0 = inst["spot"]
            payoff = inst["qty"] * (S_plot - S0)
        else:
            if main_spot != 0:
                mapped_S = S_plot * (inst["spot"]/main_spot)
            else:
                mapped_S = S_plot
            payoff = option_payoff_vector(mapped_S, inst["strike"], inst["option_type"], inst["qty"])
        total_payoff += payoff

    if st.checkbox("Show Delta hedge payoff"):
        total_payoff += -hedge_shares*(S_plot-main_spot)

    if st.checkbox("Show Gamma hedge payoff"):
        if gamma_per_contract != 0 and np.isfinite(qty_needed):
            hedge_payoff = option_payoff_vector(S_plot, cand_strike, cand_type, int(np.round(qty_needed)))
            total_payoff += hedge_payoff

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(S_plot, total_payoff, label="Portfolio P&L")
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(main_spot, color="blue", linestyle=":", label=f"Spot {main_under}={main_spot:.2f}")
    ax.set_xlabel(f"{main_under} Price at expiry")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------------------------------------------------------
    # Detailed Greeks
    # -----------------------------------------------------------------------------
    st.subheader("Instrument Greeks (scaled)")
    st.table(pd.DataFrame(instruments_info).set_index("label").style.format("{:.3f}"))

    # -----------------------------------------------------------------------------
    # Pedagogical Notes
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Pedagogical Notes")
    st.markdown("""
    - The hedging workflow follows section 7 of *Mastering the Greeks*:  
      **first neutralize Delta, then consider Gamma**.  
    - **Delta hedge**: adjust stock holdings to make total delta ≈ 0.  
    - **Gamma hedge**: use another option to reduce sensitivity of delta drift.  
    - Visualization assumes 1-factor mapping when multiple underlyings are present.  
      For real portfolios, joint simulations and correlations are required.  
    - After each hedge, recalc Greeks: hedges alter exposures (especially Gamma & Vega).  
    """)
else:
    st.info("Add instruments or select a preset to see Greeks and hedging suggestions.")
