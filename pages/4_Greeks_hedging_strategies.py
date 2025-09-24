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
st.markdown("This interactive page helps you **build a multi-asset portfolio** (stocks + options), inspect aggregated Greeks, and experiment with **Delta hedging** and **Gamma hedging** steps. Inspired by the practical notes in *Mastering the Greeks* (see section 7). :contentReference[oaicite:2]{index=2}")

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
    # qty_contracts = number of option contracts (1 contract = 100 shares)
    mult = qty_contracts * 100.0
    if option_type == "call":
        return np.maximum(S_range - strike, 0.0) * mult
    else:
        return np.maximum(strike - S_range, 0.0) * mult

def stock_payoff_vector(S_range, qty_shares):
    return qty_shares * (S_range)

# -----------------------------------------------------------------------------
# Portfolio builder (session state)
# -----------------------------------------------------------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []  # list of dicts

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
    opt_qty = st.sidebar.number_input("Quantity (number of contracts)", min_value=0, value=1, step=1, key="opt_qty")
    opt_vol = st.sidebar.number_input("Implied vol (annual, e.g. 0.2)", min_value=0.01, value=0.20, step=0.01, key="opt_vol")
    if st.sidebar.button("Add option"):
        spot = fetch_spot(opt_under)
        # compute strike based on mode
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

# Allow removing last instrument
if st.sidebar.button("Clear portfolio"):
    st.session_state.portfolio = []
    st.sidebar.info("Portfolio cleared")

# -----------------------------------------------------------------------------
# Portfolio display & composition
# -----------------------------------------------------------------------------
st.subheader("Portfolio composition")
if not st.session_state.portfolio:
    st.info("Portfolio is empty — add stocks or options from the sidebar.")
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
# Risk params (global)
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Global market params")
r = st.sidebar.number_input("Risk-free rate (r, annual)", value=0.02, step=0.005)
base_range_pct = st.sidebar.slider("Price sweep around spot (%)", 10, 100, 30)
st.sidebar.markdown("Helper: pick a price range for Greeks/payoff plots")

# -----------------------------------------------------------------------------
# Compute aggregated Greeks & payoffs
# -----------------------------------------------------------------------------
if st.session_state.portfolio:
    st.subheader("Aggregated Greeks & Hedging suggestions")
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    # Build tickers list for S_range plotting per underlying
    under_spots = {}
    for inst in st.session_state.portfolio:
        if inst["type"] == "stock":
            under_spots[inst["ticker"]] = inst.get("spot", fetch_spot(inst["ticker"]))
        else:
            u = inst["underlying"]
            under_spots[u] = inst.get("spot", fetch_spot(u))

    # Prepare S_range aggregated by a single scaled axis: we create a union of underlying spots
    # For payoff plotting we choose a representative S_range per underlying later; for aggregated Greeks plot we aggregate per-asset onto each underlying's own S_range.
    agg_payoff = None
    payoff_plots = []

    # We'll compute per-instrument Greeks and store per-instrument info
    instruments_info = []
    for inst in st.session_state.portfolio:
        if inst["type"] == "stock":
            S_i = inst.get("spot", fetch_spot(inst["ticker"]))
            qty = inst["qty"]
            # stock contributions: delta = qty, gamma/vega/theta/rho = 0
            d = qty
            g = 0.0
            v = 0.0
            t_ = 0.0
            r_ = 0.0
            total_delta += d
            total_gamma += g
            total_vega += v
            total_theta += t_
            total_rho += r_
            instruments_info.append({
                "label": f"Stock {inst['ticker']}",
                "type": "stock",
                "spot": S_i,
                "qty": qty,
                "delta": d, "gamma": g, "vega": v, "theta": t_, "rho": r_
            })
        else:
            # option
            S_i = inst.get("spot", fetch_spot(inst["underlying"]))
            K = float(inst["strike"])
            T = days_to_T(int(inst["days"]))
            sigma = float(inst["vol"])
            qty_contracts = int(inst["qty"])
            option_type = inst["option_type"]
            # greeks() returns per 1 option (not per contract) in your utils; many conventions exist.
            d, g, v, t_, r_ = greeks(S_i, K, T, r, sigma, option_type)
            # Scale: each contract is 100 shares; apply qty_contracts
            scale = qty_contracts * 100.0
            d *= scale
            g *= scale
            v *= scale
            t_ *= scale
            r_ *= scale
            total_delta += d
            total_gamma += g
            total_vega += v
            total_theta += t_
            total_rho += r_
            instruments_info.append({
                "label": f"{option_type.upper()} {inst['underlying']} K={K} x{qty_contracts}",
                "type": "option",
                "spot": S_i,
                "strike": K,
                "qty_contracts": qty_contracts,
                "vol": sigma,
                "days": inst["days"],
                "delta": d, "gamma": g, "vega": v, "theta": t_, "rho": r_
            })

    # Show totals
    totals_df = pd.DataFrame([{
        "Total Delta": total_delta,
        "Total Gamma": total_gamma,
        "Total Vega": total_vega,
        "Total Theta": total_theta,
        "Total Rho": total_rho
    }]).T
    totals_df.columns = ["Value"]
    st.table(totals_df.style.format("{:.3f}"))

    # Suggest Delta hedge (with underlying shares)
    st.markdown("### 1) Delta hedge (today)")
    st.markdown("A simple delta hedge is to trade the underlying: buy/sell `-TotalDelta` shares to neutralize today's delta.")
    hedge_shares = -total_delta
    st.info(f"To be delta-neutral today you would trade **{hedge_shares:.2f} shares** (negative = sell).")

    # Suggest Gamma hedge (using candidate option)
    st.markdown("### 2) Gamma hedge (reduce total Gamma)")
    st.markdown("Gamma hedging typically uses other options (different strikes / expiries). Below you can pick a candidate option on one underlying and compute the number of contracts required to neutralize Gamma approximately.")
    # Candidate selection
    candidate_under = st.selectbox("Choose underlying for candidate option", options=list(under_spots.keys()))
    cand_spot = under_spots[candidate_under]
    cand_strike_mode = st.selectbox("Candidate strike mode", ["ATM", "Slightly OTM (1.05x)", "Slightly ITM (0.95x)", "Custom"], key="cand_mode")
    if cand_strike_mode == "ATM":
        cand_strike = round(cand_spot, 2)
    elif cand_strike_mode == "Slightly OTM (1.05x)":
        cand_strike = round(cand_spot * 1.05, 2)
    elif cand_strike_mode == "Slightly ITM (0.95x)":
        cand_strike = round(cand_spot * 0.95, 2)
    else:
        cand_strike = st.number_input("Candidate strike (custom)", value=round(cand_spot,2), key="cand_custom_strike")
    cand_days = st.number_input("Candidate days to expiry", min_value=1, value=30, key="cand_days")
    cand_vol = st.number_input("Candidate implied vol", min_value=0.01, value=0.2, key="cand_vol")
    cand_type = st.selectbox("Candidate option type", ["call", "put"], key="cand_type")
    # compute candidate gamma per contract
    T_c = days_to_T(int(cand_days))
    d_c, g_c, v_c, t_c, r_c = greeks(cand_spot, cand_strike, T_c, r, cand_vol, cand_type)
    # gamma per contract scaled
    gamma_per_contract = g_c * 100.0
    if gamma_per_contract != 0:
        qty_contracts_needed = - total_gamma / gamma_per_contract
    else:
        qty_contracts_needed = np.nan

    st.write(f"Candidate option gamma per contract: {gamma_per_contract:.6f}")
    if np.isfinite(qty_contracts_needed):
        st.success(f"Estimated contracts to neutralize Gamma: **{qty_contracts_needed:.2f} contracts** (positive => buy)")
    else:
        st.warning("Candidate has (near) zero gamma — choose a different strike/expiry.")

    # -----------------------------------------------------------------------------
    # Payoff visualization: aggregated portfolio payoff at expiry (approx)
    # -----------------------------------------------------------------------------
    st.subheader("Payoff visualization (approx, at expiry)")
    # Build a combined S_range for visualization: use weighted average spot if multiple underlyings; but plot per-chosen underlying axis: for simplicity choose main underlying (first)
    main_under = list(under_spots.keys())[0]
    main_spot = under_spots[main_under]
    pct = base_range_pct
    S_plot = np.linspace(main_spot * (1 - pct/100), main_spot * (1 + pct/100), 200)

    total_payoff = np.zeros_like(S_plot)
    # For stock payoff, if stock underlying != main_under we map linearly by assuming correlation 1 (simplification) and scale by spot ratio -- we will warn user.
    for info, inst in zip(instruments_info, st.session_state.portfolio):
        if info["type"] == "stock":
            # if stock underlying is main_under? If multiple underlyings, payoff mapping is simplistic (assumes identical asset moves)
            qty = info["qty"]
            # stock payoff = qty * (S_plot)  -- but we want payoff relative to initial value: P&L = qty*(S_plot - S0)
            S0 = info["spot"] or main_spot
            payoff = qty * (S_plot - S0)
            total_payoff += payoff
        else:
            # option payoff: if underlying != main_under we map strike relative via spot ratio (simplified)
            strike = info["strike"]
            S0 = info["spot"] or main_spot
            # map S_plot to this underlying's price linearly: S_mapped = S_plot * (S0 / main_spot) 
            if main_spot != 0:
                S_mapped = S_plot * (S0 / main_spot)
            else:
                S_mapped = S_plot
            payoff = option_payoff_vector(S_mapped, strike, inst["option_type"], int(inst["qty"]))
            total_payoff += payoff

    # include delta hedge effect if user wants
    if st.checkbox("Show delta-hedged payoff (apply hedge_shares)"):
        # adding cash from selling/buying hedge_shares at spot: P&L from hedge = -hedge_shares * (S_plot - main_spot)
        delta_hedge_pl = - hedge_shares * (S_plot - main_spot)
        total_payoff += delta_hedge_pl

    # include gamma hedge option if user chooses
    if st.checkbox("Show Gamma hedge (use candidate option)"):
        # candidate payoff scaled by needed contracts
        if np.isfinite(qty_contracts_needed):
            candidate_payoff = option_payoff_vector(S_plot * (cand_spot / main_spot), cand_strike, cand_type, int(np.round(qty_contracts_needed)))
            total_payoff += candidate_payoff
        else:
            st.warning("Gamma hedge not plotted (invalid quantity).")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(S_plot, total_payoff, label="Portfolio P&L at expiry (approx)")
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(main_spot, color="blue", linestyle=":", label=f"Spot {main_under}={main_spot:.2f}")
    ax.set_xlabel(f"{main_under} Price at expiry")
    ax.set_ylabel("P&L ($)")
    ax.set_title("Approximate Portfolio Payoff at Expiry")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------------------------------------------------------
    # Detailed instrument Greeks table
    # -----------------------------------------------------------------------------
    st.subheader("Detailed instrument Greeks (scaled)")
    info_rows = []
    for info in instruments_info:
        info_rows.append({
            "Instrument": info["label"],
            "Spot": info.get("spot", None),
            "Delta": info["delta"],
            "Gamma": info["gamma"],
            "Vega": info["vega"],
            "Theta": info["theta"],
            "Rho": info["rho"]
        })
    st.table(pd.DataFrame(info_rows).set_index("Instrument").style.format("{:.3f}"))

    # Pedagogical notes (inspired by section 7)
    st.markdown("---")
    st.subheader("Pedagogical notes (how to use these outputs)")
    st.markdown("""
    * This page follows the practical workflow described in section 7 of *Mastering the Greeks*: 
      first neutralize Delta (today) then address Gamma to reduce hedge rebalancing frequency. :contentReference[oaicite:3]{index=3}
    * **Delta hedge**: trade the underlying to set Total Delta ≈ 0 (we show the number of shares).
    * **Gamma hedge**: add options (different strikes/expiries) to reduce Total Gamma — we give an approximate number of contracts for a candidate option.
    * Payoff plots are approximate: when portfolio contains different underlyings we map prices linearly for visualization — for realistic multi-asset analysis you should simulate joint stochastic paths (correlations) and reprice options dynamically.
    * After implementing hedges, always re-calc Greeks because hedges change exposures (especially Gamma and Vega).
    """)

else:
    st.info("Add at least one instrument to the portfolio to see aggregated Greeks and hedging suggestions.")
