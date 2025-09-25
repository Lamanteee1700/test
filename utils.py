import streamlit as st
import numpy as np

import yfinance as yf

from scipy.stats import norm
from scipy.optimize import brentq
# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from datetime import datetime, timedelta

# --- SHARED CONSTANTS ---
TRADING_DAYS_YEAR = 252
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# --- BLACK-SCHOLES CORE FUNCTIONS ---
def d1(S, K, T, r, sigma):
    """Calculate d1 parameter for Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return 0
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Calculate d2 parameter for Black-Scholes formula"""
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

def bs_price(S, K, T, r, sigma, option="call"):
    """Black-Scholes option pricing formula"""
    if T <= 0:
        # At expiration
        if option == "call":
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    if sigma <= 0:
        # No volatility case
        discounted_forward = S * np.exp(r * T)
        if option == "call":
            return max(0, discounted_forward - K) * np.exp(-r * T)
        else:
            return max(0, K - discounted_forward) * np.exp(-r * T)
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    if option == "call":
        return S*norm.cdf(d1_val) - K*np.exp(-r*T)*norm.cdf(d2_val)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2_val) - S*norm.cdf(-d1_val)

def greeks(S, K, T, r, sigma, option="call"):
    """Calculate all Greeks for an option"""
    if T <= 0:
        return 0, 0, 0, 0, 0  # All Greeks are zero at expiration
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Delta
    delta = norm.cdf(d1_val) if option=="call" else -norm.cdf(-d1_val)
    
    # Gamma  
    gamma = norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    # Vega (per 1% change in vol, divide by 100)
    vega  = S * norm.pdf(d1_val) * np.sqrt(T) / 100
    # Theta (per day)
    if option == "call":
        theta = (-(S*norm.pdf(d1_val)*sigma)/(2*np.sqrt(T)) 
                 - r*K*np.exp(-r*T)*norm.cdf(d2_val)) / 365
    else: # put
        theta = (-(S*norm.pdf(d1_val)*sigma)/(2*np.sqrt(T)) 
                 + r*K*np.exp(-r*T)*norm.cdf(-d2_val)) / 365
    
    # Rho (per 1% change in rates, divide by 100)
    if option == "call":
        rho   = (K*T*np.exp(-r*T)*norm.cdf(d2_val)) / 100
    else: # put
        rho   = (-K*T*np.exp(-r*T)*norm.cdf(-d2_val)) / 100

    return delta, gamma, vega, theta, rho
