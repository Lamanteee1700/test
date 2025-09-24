import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Black-Scholes Helper Functions ---
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

# --- Black-Scholes Option Pricing ---
def bs_price(S, K, T, r, sigma, option="call"):
    """Returns the Black-Scholes price of a call or put option."""
    if option == "call":
        return S*norm.cdf(d1(S,K,T,r,sigma)) - K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)) - S*norm.cdf(-d1(S,K,T,r,sigma))

# --- Greeks ---
def greeks(S, K, T, r, sigma, option="call"):
    """Returns the Greeks (Delta, Gamma, Vega, Theta, Rho)."""
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    # Delta
    delta = norm.cdf(d1_val) if option=="call" else -norm.cdf(-d1_val)
    # Gamma
    gamma = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    # Vega (per 1% change in vol, divide by 100)
    vega  = S * norm.pdf(d1_val) * np.sqrt(T) / 100
    # Theta (per day)
    theta = (-(S*norm.pdf(d1_val)*sigma)/(2*np.sqrt(T)) 
             - r*K*np.exp(-r*T)*norm.cdf(d2_val if option=="call" else -d2_val)) / 365
    # Rho (per 1% change in rates, divide by 100)
    rho   = (K*T*np.exp(-r*T)*norm.cdf(d2_val if option=="call" else -d2_val)) / 100

    return delta, gamma, vega, theta, rho
