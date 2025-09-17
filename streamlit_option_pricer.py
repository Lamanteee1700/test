# streamlit_option_pricer.py
# Simple, single-file Streamlit app that prices European options
# using Black-Scholes, Monte Carlo simulation, and a CRR binomial tree.
# Works if only streamlit is installed; numpy is used if available for speed.
# Run: streamlit run streamlit_option_pricer.py

import streamlit as st
from math import log, sqrt, exp, pi, erf
import random
from datetime import date

# Try to import numpy for speed; fall back to pure-python if not available
try:
    import numpy as np
except Exception:
    np = None

st.set_page_config(page_title="Options pricer", layout="wide")
st.title("🧮 Options pricing playground — Black‑Scholes / Monte Carlo / Binomial")
st.markdown("""
This app computes European option prices (call/put) with:
- Black–Scholes formula (analytic) — shows Greeks
- Monte Carlo (log-normal simulation)
- Binomial (Cox–Ross–Rubinstein)

Tips: adjust time-to-expiry in days, enter rates as annual percentages, and volatility as annual %.
""")

# -------------------- Sidebar: parameters --------------------
with st.sidebar:
    st.header("Option inputs")
    option_type = st.selectbox("Option type", ["Call", "Put"])  # display
    S = st.number_input("Spot price (S)", min_value=0.0, value=100.0, step=1.0, format="%f")
    K = st.number_input("Strike price (K)", min_value=0.0, value=100.0, step=1.0, format="%f")

    days = st.number_input("Time to expiry — days", min_value=0, value=30, step=1)
    T = float(days) / 365.0

    r_pct = st.number_input("Risk‑free rate (annual %, r)", value=1.0, step=0.1)
    q_pct = st.number_input("Dividend yield (annual %, q)", value=0.0, step=0.1)
    sigma_pct = st.number_input("Volatility (annual %, sigma)", value=20.0, step=0.1)

    r = float(r_pct) / 100.0
    q = float(q_pct) / 100.0
    sigma = float(sigma_pct) / 100.0

    st.markdown("---")
    st.header("Pricing method")
    method = st.selectbox("Method", ["Black‑Scholes (analytic)", "Monte Carlo", "Binomial (CRR)"])

    # method-specific inputs
    if method == "Monte Carlo":
        mc_paths = st.number_input("MC paths (simulations)", min_value=100, value=20000, step=100)
        mc_seed = st.number_input("Random seed (0 for random)", min_value=0, value=0, step=1)
    if method == "Binomial (CRR)":
        binom_steps = st.number_input("Binomial steps", min_value=1, value=200, step=1)

    st.markdown("---")
    st.header("Implied volatility")
    market_price_input = st.number_input("Market option price (leave 0 to skip)", min_value=0.0, value=0.0, step=0.01, format="%f")

    st.markdown("---")
    st.caption("Note: if numpy is not installed the app will still run but simulations are slower.")

# -------------------- Math helpers --------------------

def norm_cdf(x):
    """Standard normal CDF using math.erf (accurate enough).
    Vectorized if numpy is present.
    """
    if np is not None:
        return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    if np is not None:
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

# -------------------- Pricing formulas --------------------

def bs_price(S, K, T, r, q, sigma, option_type="call"):
    # Handle immediate expiry
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:
        return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0:
        # at expiry Greeks collapse to simple quantities (approx)
        intrinsic = (S - K) if option_type == "call" else (K - S)
        delta = 1.0 if (intrinsic > 0 and option_type == "call") else (0.0 if option_type == "call" else -1.0 if intrinsic > 0 else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    pdf_d1 = norm_pdf(d1)

    delta_call = exp(-q * T) * norm_cdf(d1)
    delta_put = exp(-q * T) * (norm_cdf(d1) - 1.0)
    delta = delta_call if option_type == "call" else delta_put

    gamma = exp(-q * T) * pdf_d1 / (S * sigma * sqrt(T))
    vega = S * exp(-q * T) * pdf_d1 * sqrt(T)

    theta_call = - (S * pdf_d1 * sigma * exp(-q * T)) / (2.0 * sqrt(T)) - r * K * exp(-r * T) * norm_cdf(d2) + q * S * exp(-q * T) * norm_cdf(d1)
    theta_put = - (S * pdf_d1 * sigma * exp(-q * T)) / (2.0 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(-d2) - q * S * exp(-q * T) * norm_cdf(-d1)
    theta = theta_call if option_type == "call" else theta_put

    rho_call = K * T * exp(-r * T) * norm_cdf(d2)
    rho_put = -K * T * exp(-r * T) * norm_cdf(-d2)
    rho = rho_call if option_type == "call" else rho_put

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

# -------------------- Implied vol (bisection) --------------------

def implied_vol_bisect(market_price, S, K, T, r, q, option_type="call", tol=1e-6, max_iter=200):
    # trivial bounds
    if market_price <= 0:
        return 0.0

    # price bounds: intrinsic <= price <= S*exp(-qT) (call) or K*exp(-rT) (put)
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    upper = max(S * exp(-q * T), K * exp(-r * T)) * 2.0

    low = 1e-8
    high = 5.0
    for i in range(max_iter):
        mid = 0.5 * (low + high)
        p = bs_price(S, K, T, r, q, mid, option_type)
        # adjust
        if abs(p - market_price) < tol:
            return mid
        if p > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# -------------------- Monte Carlo pricing --------------------

def mc_price(S, K, T, r, q, sigma, option_type="call", paths=20000, seed=0):
    if T <= 0:
        return (max(0.0, S - K) if option_type == "call" else max(0.0, K - S)), [max(0.0, S - K) if option_type == "call" else max(0.0, K - S)]

    if np is not None:
        if seed != 0:
            np.random.seed(int(seed))
        Z = np.random.normal(size=int(paths))
        ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z)
        if option_type == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        price = exp(-r * T) * float(np.mean(payoff))
        return price, payoff.tolist()
    else:
        # fallback pure-python
        if seed != 0:
            random.seed(int(seed))
        payoffs = []
        drift = (r - q - 0.5 * sigma * sigma) * T
        vol_sqrt = sigma * sqrt(T)
        for i in range(int(paths)):
            Z = random.gauss(0.0, 1.0)
            ST = S * exp(drift + vol_sqrt * Z)
            if option_type == "call":
                payoffs.append(max(0.0, ST - K))
            else:
                payoffs.append(max(0.0, K - ST))
        price = exp(-r * T) * (sum(payoffs) / float(paths))
        return price, payoffs

# -------------------- Binomial CRR pricing --------------------

def binomial_crr(S, K, T, r, q, sigma, option_type="call", steps=200):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    N = int(steps)
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    a = exp((r - q) * dt)
    p = (a - d) / (u - d)

    # terminal payoffs
    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    if option_type == "call":
        payoffs = [max(0.0, p_ - K) for p_ in prices]
    else:
        payoffs = [max(0.0, K - p_) for p_ in prices]

    # backward induction
    for i in range(N, 0, -1):
        payoffs = [exp(-r * dt) * (p * payoffs[j + 1] + (1 - p) * payoffs[j]) for j in range(i)]
    return payoffs[0]

# -------------------- UI: compute and present --------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs summary")
    st.write({
        "Option": option_type,
        "S": S,
        "K": K,
        "T (years)": round(T, 6),
        "r": r,
        "q": q,
        "sigma": sigma,
        "Method": method,
    })

    st.markdown("---")
    if st.button("Price option"):
        method_chosen = method
        ot = option_type.lower()
        if method_chosen == "Black‑Scholes (analytic)":
            price = bs_price(S, K, T, r, q, sigma, ot)
            greeks = bs_greeks(S, K, T, r, q, sigma, ot)
            st.success(f"Price (Black–Scholes): {price:.6f}")
            st.write("Greeks:")
            st.write({k: (v if abs(v) >= 1e-12 else 0.0) for k, v in greeks.items()})
        elif method_chosen == "Monte Carlo":
            paths = int(mc_paths)
            seed = int(mc_seed)
            price, payoffs = mc_price(S, K, T, r, q, sigma, ot, paths=paths, seed=seed)
            st.success(f"Price (Monte Carlo, {paths} paths): {price:.6f}")
            # stats
            mean_payoff = sum(payoffs) / len(payoffs)
            st.write({"discounted mean payoff": exp(-r * T) * mean_payoff, "raw mean payoff": mean_payoff})
            # histogram
            bins = 40
            if np is not None:
                hist, edges = np.histogram(payoffs, bins=bins)
                st.subheader("Monte Carlo payoff histogram (raw payoffs)")
                st.bar_chart(hist)
            else:
                # simple python histogram
                minp = min(payoffs)
                maxp = max(payoffs)
                if maxp == minp:
                    st.write("All payoffs identical — no histogram to show.")
                else:
                    bin_counts = [0] * bins
                    for v in payoffs:
                        idx = int((v - minp) / (maxp - minp + 1e-12) * (bins - 1))
                        bin_counts[idx] += 1
                    st.subheader("Monte Carlo payoff histogram (counts)")
                    st.bar_chart(bin_counts)
        else:  # binomial
            steps = int(binom_steps)
            price = binomial_crr(S, K, T, r, q, sigma, ot, steps=steps)
            st.success(f"Price (Binomial CRR, {steps} steps): {price:.6f}")

# If implied vol requested
if market_price_input > 0.0:
    ot = option_type.lower()
    implied = implied_vol_bisect(market_price_input, S, K, T, r, q, ot)
    st.subheader("Implied volatility (bisection)")
    st.write({"market_price": market_price_input, "implied_vol": implied, "implied_vol_%": implied * 100.0})

# Main column: payoff charts and exploratory plots
with col2:
    st.subheader("Payoff at expiry")
    # payoff vs underlying price
    s_min = max(0.0, S * 0.5)
    s_max = S * 1.5 + 1e-6
    grid = None
    if np is not None:
        grid = np.linspace(s_min, s_max, 201)
        if option_type == "Call":
            payoff = np.maximum(grid - K, 0.0)
        else:
            payoff = np.maximum(K - grid, 0.0)
        st.line_chart({"underlying": grid.tolist(), "payoff": payoff.tolist()})
    else:
        grid = [s_min + i * (s_max - s_min) / 200.0 for i in range(201)]
        if option_type == "Call":
            payoff = [max(0.0, s_ - K) for s_ in grid]
        else:
            payoff = [max(0.0, K - s_) for s_ in grid]
        st.line_chart({"underlying": grid, "payoff": payoff})

    st.markdown("---")
    st.subheader("Extra experiments")
    # Price vs volatility chart
    vol_grid = [max(0.001, sigma * factor) for factor in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
    prices_vs_vol = [bs_price(S, K, T, r, q, v, option_type.lower()) for v in vol_grid]
    st.write("Price vs volatility (sample points)")
    st.line_chart({"vol": [v * 100.0 for v in vol_grid], "price": prices_vs_vol})

st.markdown("---")
st.caption("This is a teaching tool. For production use, add input validation, tests, and vectorized computation.")
