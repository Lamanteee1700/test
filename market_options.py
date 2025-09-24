import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def get_options_data(ticker):
    """
    Fetch options data for a given ticker using Yahoo Finance
    
    Returns:
    - options_data: Dict containing calls/puts data for different expirations
    - expiration_dates: List of available expiration dates
    - current_price: Current stock price
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get current stock price
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        # Get available expiration dates
        expiration_dates = stock.options
        
        if not expiration_dates:
            return None, None, current_price
        
        options_data = {}
        
        # Fetch options data for each expiration date (limit to first 6 for performance)
        for exp_date in expiration_dates[:6]:
            try:
                option_chain = stock.option_chain(exp_date)
                
                # Process calls and puts
                calls_df = option_chain.calls
                puts_df = option_chain.puts
                
                # Add useful calculated fields
                if not calls_df.empty:
                    calls_df['mid_price'] = (calls_df['bid'] + calls_df['ask']) / 2
                    calls_df['spread'] = calls_df['ask'] - calls_df['bid']
                    calls_df['spread_pct'] = calls_df['spread'] / calls_df['mid_price'] * 100
                    calls_df['moneyness'] = current_price / calls_df['strike']
                
                if not puts_df.empty:
                    puts_df['mid_price'] = (puts_df['bid'] + puts_df['ask']) / 2
                    puts_df['spread'] = puts_df['ask'] - puts_df['bid']
                    puts_df['spread_pct'] = puts_df['spread'] / puts_df['mid_price'] * 100
                    puts_df['moneyness'] = current_price / puts_df['strike']
                
                options_data[exp_date] = {
                    'calls': calls_df,
                    'puts': puts_df,
                    'expiration': exp_date
                }
                
            except Exception as e:
                st.warning(f"Could not fetch options for {exp_date}: {str(e)}")
                continue
        
        return options_data, expiration_dates, current_price
        
    except Exception as e:
        st.error(f"Error fetching options data for {ticker}: {str(e)}")
        return None, None, None

def find_closest_strike(options_df, target_strike):
    """Find the option with strike closest to target"""
    if options_df.empty:
        return None
    
    # Find closest strike
    strike_diff = abs(options_df['strike'] - target_strike)
    closest_idx = strike_diff.idxmin()
    
    return options_df.loc[closest_idx]

def get_market_option_price(ticker, strike, expiration_date, option_type='call'):
    """
    Get market price for a specific option
    
    Parameters:
    - ticker: Stock ticker symbol
    - strike: Option strike price
    - expiration_date: Expiration date string (YYYY-MM-DD)
    - option_type: 'call' or 'put'
    
    Returns:
    - Dictionary with bid, ask, mid prices and other market data
    """
    try:
        stock = yf.Ticker(ticker)
        option_chain = stock.option_chain(expiration_date)
        
        if option_type.lower() == 'call':
            options_df = option_chain.calls
        else:
            options_df = option_chain.puts
        
        if options_df.empty:
            return None
        
        # Find exact strike or closest
        exact_strike = options_df[options_df['strike'] == strike]
        
        if not exact_strike.empty:
            option_data = exact_strike.iloc[0]
        else:
            # Find closest strike
            option_data = find_closest_strike(options_df, strike)
            if option_data is None:
                return None
        
        return {
            'strike': option_data['strike'],
            'bid': option_data['bid'],
            'ask': option_data['ask'],
            'mid_price': (option_data['bid'] + option_data['ask']) / 2,
            'last_price': option_data['lastPrice'],
            'volume': option_data['volume'],
            'open_interest': option_data['openInterest'],
            'implied_volatility': option_data['impliedVolatility'],
            'spread': option_data['ask'] - option_data['bid'],
            'spread_pct': ((option_data['ask'] - option_data['bid']) / 
                          ((option_data['bid'] + option_data['ask']) / 2)) * 100 
                          if (option_data['bid'] + option_data['ask']) > 0 else 0
        }
        
    except Exception as e:
        st.error(f"Error fetching market price: {str(e)}")
        return None

def display_options_chain(ticker, max_expirations=3):
    """Display options chain data in Streamlit"""
    
    st.subheader(f"📊 Live Options Chain: {ticker}")
    
    # Fetch options data
    with st.spinner("Fetching live options data..."):
        options_data, expiration_dates, current_price = get_options_data(ticker)
    
    if not options_data:
        st.error("No options data available for this ticker")
        return None, None
    
    st.success(f"Current Stock Price: ${current_price:.2f}")
    
    # Expiration date selector
    exp_date = st.selectbox(
        "Select Expiration Date",
        list(options_data.keys())[:max_expirations],
        help="Choose from available expiration dates"
    )
    
    if exp_date not in options_data:
        return None, None
    
    # Calculate days to expiration
    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
    days_to_exp = (exp_datetime - datetime.now()).days
    
    st.info(f"Selected Expiration: {exp_date} ({days_to_exp} days)")
    
    # Display options chain
    calls_df = options_data[exp_date]['calls']
    puts_df = options_data[exp_date]['puts']
    
    # Filter for reasonable strikes (± 20% from current price)
    price_range = 0.3  # 30% range
    min_strike = current_price * (1 - price_range)
    max_strike = current_price * (1 + price_range)
    
    calls_filtered = calls_df[
        (calls_df['strike'] >= min_strike) & 
        (calls_df['strike'] <= max_strike)
    ].copy()
    
    puts_filtered = puts_df[
        (puts_df['strike'] >= min_strike) & 
        (puts_df['strike'] <= max_strike)
    ].copy()
    
    # Display options tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 CALLS**")
        if not calls_filtered.empty:
            display_cols = ['strike', 'bid', 'ask', 'mid_price', 'volume', 
                          'openInterest', 'impliedVolatility']
            calls_display = calls_filtered[display_cols].round(3)
            calls_display.columns = ['Strike', 'Bid', 'Ask', 'Mid', 'Volume', 
                                   'OI', 'IV']
            st.dataframe(calls_display, use_container_width=True)
        else:
            st.write("No call options in price range")
    
    with col2:
        st.write("**📉 PUTS**") 
        if not puts_filtered.empty:
            display_cols = ['strike', 'bid', 'ask', 'mid_price', 'volume', 
                          'openInterest', 'impliedVolatility']
            puts_display = puts_filtered[display_cols].round(3)
            puts_display.columns = ['Strike', 'Bid', 'Ask', 'Mid', 'Volume', 
                                   'OI', 'IV']
            st.dataframe(puts_display, use_container_width=True)
        else:
            st.write("No put options in price range")
    
    return options_data[exp_date], current_price

def create_iv_smile_chart(calls_df, puts_df, current_price):
    """Create implied volatility smile chart"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    if not calls_df.empty:
        # Filter for liquid options (volume > 0 or open interest > 10)
        liquid_calls = calls_df[
            (calls_df['volume'] > 0) | (calls_df['openInterest'] > 10)
        ].copy()
        
        if not liquid_calls.empty:
            liquid_calls['moneyness'] = liquid_calls['strike'] / current_price
            fig.add_trace(go.Scatter(
                x=liquid_calls['moneyness'],
                y=liquid_calls['impliedVolatility'] * 100,
                mode='markers+lines',
                name='Calls IV',
                marker=dict(color='green', size=6)
            ))
    
    if not puts_df.empty:
        liquid_puts = puts_df[
            (puts_df['volume'] > 0) | (puts_df['openInterest'] > 10)
        ].copy()
        
        if not liquid_puts.empty:
            liquid_puts['moneyness'] = liquid_puts['strike'] / current_price
            fig.add_trace(go.Scatter(
                x=liquid_puts['moneyness'],
                y=liquid_puts['impliedVolatility'] * 100,
                mode='markers+lines',
                name='Puts IV',
                marker=dict(color='red', size=6)
            ))
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="black", 
                  annotation_text="ATM")
    
    fig.update_layout(
        title="Implied Volatility Smile",
        xaxis_title="Moneyness (Strike/Spot)",
        yaxis_title="Implied Volatility (%)",
        height=400
    )
    
    return fig

# Integration function for the options pricer
def integrate_market_data_fetcher():
    """
    Function to integrate market data fetching into the options pricer
    """
    st.subheader("📊 Live Market Data Integration")
    
    use_market_data = st.checkbox(
        "Use Live Market Option Prices", 
        help="Fetch real option prices from the market instead of manual input"
    )
    
    if use_market_data:
        st.info("""
        **Market Data Features:**
        - ✅ Real-time bid/ask/mid prices
        - ✅ Implied volatility from market
        - ✅ Volume and open interest data  
        - ✅ Multiple expiration dates
        - ✅ Automatic strike matching
        - ⚠️ Limited to liquid options
        - ⚠️ US equities only (Yahoo Finance)
        - ⚠️ 15-20 minute delay on free data
        """)
        
        return True
    
    return False
