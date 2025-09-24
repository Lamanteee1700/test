# pages/12_Fixed_income_fundamentals.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def bond_price(face_value, coupon_rate, yield_rate, periods):
    """Calculate bond price using present value formula"""
    coupon = face_value * coupon_rate
    pv_coupons = sum([coupon / (1 + yield_rate)**t for t in range(1, periods + 1)])
    pv_principal = face_value / (1 + yield_rate)**periods
    return pv_coupons + pv_principal

def bond_duration(face_value, coupon_rate, yield_rate, periods):
    """Calculate Macaulay duration"""
    coupon = face_value * coupon_rate
    bond_price_val = bond_price(face_value, coupon_rate, yield_rate, periods)
    
    weighted_time = 0
    for t in range(1, periods + 1):
        if t < periods:
            cash_flow = coupon
        else:
            cash_flow = coupon + face_value
        
        pv_cash_flow = cash_flow / (1 + yield_rate)**t
        weighted_time += (t * pv_cash_flow)
    
    return weighted_time / bond_price_val

def modified_duration(macaulay_duration, yield_rate):
    """Calculate modified duration"""
    return macaulay_duration / (1 + yield_rate)

def bond_convexity(face_value, coupon_rate, yield_rate, periods):
    """Calculate bond convexity"""
    coupon = face_value * coupon_rate
    bond_price_val = bond_price(face_value, coupon_rate, yield_rate, periods)
    
    convexity_sum = 0
    for t in range(1, periods + 1):
        if t < periods:
            cash_flow = coupon
        else:
            cash_flow = coupon + face_value
        
        pv_cash_flow = cash_flow / (1 + yield_rate)**t
        convexity_sum += (t * (t + 1) * pv_cash_flow)
    
    return convexity_sum / (bond_price_val * (1 + yield_rate)**2)

def create_yield_curve_data():
    """Generate sample yield curve data"""
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Normal yield curve
    normal_curve = 0.02 + 0.003 * np.log(1 + maturities) + 0.0005 * maturities
    
    # Inverted yield curve
    inverted_curve = 0.045 - 0.01 * np.log(1 + maturities)
    
    # Flat yield curve
    flat_curve = np.full_like(maturities, 0.035)
    
    return maturities, normal_curve, inverted_curve, flat_curve

def simulate_interest_rate_scenarios():
    """Simulate different interest rate scenarios"""
    np.random.seed(42)
    
    # Base scenario parameters
    initial_rate = 0.03
    periods = 120  # 10 years monthly
    
    scenarios = {}
    
    # Scenario 1: Gradual rise
    scenarios['Gradual Rise'] = [initial_rate + 0.0002 * t + 0.001 * np.random.randn() for t in range(periods)]
    
    # Scenario 2: Sharp rise then decline
    sharp_rise = [initial_rate + max(0, 0.001 * (t - 60)**2 / 1000) + 0.001 * np.random.randn() for t in range(periods)]
    scenarios['Shock Rise'] = sharp_rise
    
    # Scenario 3: Volatile sideways
    scenarios['Volatile Sideways'] = [initial_rate + 0.005 * np.random.randn() for _ in range(periods)]
    
    return scenarios

def calculate_credit_spread_history():
    """Generate sample credit spread data"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    
    # Investment grade spreads (basis points)
    ig_base = 100
    ig_spreads = []
    
    # High yield spreads (basis points)  
    hy_base = 400
    hy_spreads = []
    
    for i, date in enumerate(dates):
        # Add cyclicality and crisis effects
        cycle_factor = 1 + 0.3 * np.sin(i * 0.1)
        
        # COVID crisis effect (2020)
        if date.year == 2020 and date.month >= 3:
            crisis_factor = 1.8
        else:
            crisis_factor = 1.0
        
        ig_spread = ig_base * cycle_factor * crisis_factor + np.random.normal(0, 10)
        hy_spread = hy_base * cycle_factor * crisis_factor + np.random.normal(0, 30)
        
        ig_spreads.append(max(50, ig_spread))  # Floor at 50bps
        hy_spreads.append(max(200, hy_spread))  # Floor at 200bps
    
    return dates, ig_spreads, hy_spreads

def show_fixed_income_page():
    st.set_page_config(page_title="Fixed Income Fundamentals", layout="wide")
    
    st.title("🏦 Fixed Income Fundamentals: The Backbone of Financial Markets")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Understanding Bonds, Yields, and Interest Rate Risk</h3>
        <p style="color: #e8f4fd; margin: 0.5rem 0 0 0;">
            Master the fundamentals of fixed income securities from basic bond mechanics to advanced duration and convexity concepts
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === SECTION 1: WHAT IS FIXED INCOME? ===
    st.subheader("📊 What is Fixed Income?")
    
    with st.expander("🎯 Definition and Key Characteristics", expanded=True):
        st.markdown("""
        **Fixed income** refers to investment securities that provide predictable cash flows over a defined period, 
        typically through periodic interest payments and return of principal at maturity.
        
        ### Core Components of a Bond:
        
        **1. Par Value (Face Value)**
        - The amount repaid to investors at maturity
        - Typically $1,000 for corporate bonds, $100 for government bonds
        - Market prices are quoted as percentage of par value
        
        **2. Coupon Rate**
        - Annual interest rate applied to par value
        - Determines periodic interest payments
        - Can be fixed or floating (variable)
        
        **3. Maturity Date**
        - When the issuer must repay the principal
        - Ranges from days (Treasury bills) to decades (long bonds)
        - Affects interest rate sensitivity
        
        **4. Credit Quality**
        - Assessment of issuer's ability to meet obligations
        - Reflected in credit ratings (AAA, AA, A, BBB, etc.)
        - Influences required yield and risk premium
        
        ### Why "Fixed" Income?
        
        The term originates from the contractual nature of payments - investors know in advance 
        the amount and timing of cash flows if the issuer meets obligations. However, not all 
        "fixed income" has truly fixed payments:
        
        - **Floating Rate Notes**: Coupons adjust with benchmark rates
        - **Inflation-Linked Bonds**: Payments adjust for inflation changes  
        - **Asset-Backed Securities**: Payments depend on underlying assets
        """)
    
    # === SECTION 2: BOND PRICING FUNDAMENTALS ===
    st.subheader("💰 Bond Pricing: Present Value of Future Cash Flows")
    
    st.markdown("""
    **Bond pricing** is based on the fundamental principle that a bond's value equals the present value 
    of its future cash flows, discounted at the appropriate yield (required rate of return).
    """)
    
    # Interactive Bond Pricer
    with st.container():
        st.markdown("#### Interactive Bond Pricing Calculator")
        
        pricing_col1, pricing_col2 = st.columns(2)
        
        with pricing_col1:
            st.markdown("**Bond Parameters**")
            
            face_value = st.number_input("Par Value ($)", value=1000, min_value=100, step=100)
            coupon_rate = st.slider("Annual Coupon Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100
            years_to_maturity = st.slider("Years to Maturity", 1, 30, 10)
            market_yield = st.slider("Market Yield (YTM) (%)", 0.0, 12.0, 5.0, 0.25) / 100
        
        with pricing_col2:
            st.markdown("**Calculated Values**")
            
            # Calculate bond metrics
            bond_price_val = bond_price(face_value, coupon_rate, market_yield, years_to_maturity)
            mac_duration = bond_duration(face_value, coupon_rate, market_yield, years_to_maturity)
            mod_duration = modified_duration(mac_duration, market_yield)
            convexity = bond_convexity(face_value, coupon_rate, market_yield, years_to_maturity)
            
            annual_coupon = face_value * coupon_rate
            current_yield = annual_coupon / bond_price_val if bond_price_val > 0 else 0
            
            # Display results
            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Bond Price", f"${bond_price_val:.2f}")
            result_col2.metric("Price vs Par", f"{bond_price_val/face_value:.1%}", 
                             "Premium" if bond_price_val > face_value else "Discount" if bond_price_val < face_value else "At Par")
            result_col1.metric("Current Yield", f"{current_yield:.2%}")
            result_col2.metric("Yield to Maturity", f"{market_yield:.2%}")
    
    # Price-Yield Relationship Chart
    st.markdown("#### The Inverse Price-Yield Relationship")
    
    # Generate price-yield curve
    yield_range = np.linspace(0.01, 0.12, 50)
    prices = [bond_price(face_value, coupon_rate, y, years_to_maturity) for y in yield_range]
    
    fig_price_yield = go.Figure()
    fig_price_yield.add_trace(go.Scatter(
        x=yield_range*100, y=prices,
        mode='lines',
        name='Bond Price',
        line=dict(color='blue', width=3)
    ))
    
    # Mark current position
    fig_price_yield.add_trace(go.Scatter(
        x=[market_yield*100], y=[bond_price_val],
        mode='markers',
        name='Current Position',
        marker=dict(size=12, color='red', symbol='diamond')
    ))
    
    fig_price_yield.add_hline(y=face_value, line_dash="dash", line_color="gray", 
                             annotation_text="Par Value")
    
    fig_price_yield.update_layout(
        title=f"Price-Yield Relationship: {coupon_rate:.1%} Coupon, {years_to_maturity}Y Maturity",
        xaxis_title="Yield to Maturity (%)",
        yaxis_title="Bond Price ($)",
        height=400
    )
    
    st.plotly_chart(fig_price_yield, use_container_width=True)
    
    with st.expander("📖 Understanding the Price-Yield Relationship"):
        st.markdown("""
        ### The Inverse Relationship Explained
        
        The **inverse relationship** between bond prices and yields is fundamental to fixed income:
        
        **When yields rise → Bond prices fall**  
        **When yields fall → Bond prices rise**
        
        **Why does this happen?**
        
        1. **Fixed Cash Flows**: Bond coupon payments are typically fixed at issuance
        2. **Competitive Returns**: New bonds must offer competitive yields to attract investors
        3. **Present Value Impact**: Higher discount rates reduce the present value of future cash flows
        
        **Mathematical Foundation:**
        
        ```
        Bond Price = Σ[Coupon/(1+YTM)^t] + Par Value/(1+YTM)^n
        ```
        
        **Key Observations:**
        
        - **Convex Shape**: The curve is convex (bowed), not linear
        - **Asymmetric Impact**: Price gains from yield drops exceed losses from yield rises
        - **Maturity Effect**: Longer maturity bonds show greater price sensitivity
        - **Coupon Effect**: Lower coupon bonds are more sensitive to yield changes
        
        **Practical Implications:**
        
        - **Rising Rate Environment**: Bond prices decline, but new investments earn higher yields
        - **Falling Rate Environment**: Bond prices rise, providing capital gains
        - **Duration Risk**: Longer duration bonds have higher price volatility
        """)
    
    # === SECTION 3: DURATION AND CONVEXITY ===
    st.subheader("⏱️ Duration and Convexity: Measuring Interest Rate Risk")
    
    st.markdown("""
    **Duration** measures a bond's price sensitivity to changes in yield. It represents the weighted average 
    time until cash flows are received and provides a linear approximation of price changes for small yield movements.
    
    **Convexity** captures the curvature of the price-yield relationship, improving duration estimates for larger yield changes.
    """)
    
    # Duration and Convexity Display
    with st.container():
        duration_col1, duration_col2 = st.columns(2)
        
        with duration_col1:
            st.markdown("**Duration Measures**")
            st.metric("Macaulay Duration", f"{mac_duration:.2f} years", 
                     help="Weighted average time to receive cash flows")
            st.metric("Modified Duration", f"{mod_duration:.2f}", 
                     help="Price sensitivity per 1% yield change")
            
            # Duration interpretation
            price_change_1pct = -mod_duration * 0.01 * 100  # 1% yield change
            st.info(f"**Duration Effect**: For a 1% yield increase, bond price would decline by approximately {abs(price_change_1pct):.1f}%")
        
        with duration_col2:
            st.markdown("**Convexity Analysis**")
            st.metric("Convexity", f"{convexity:.2f}", 
                     help="Curvature of price-yield relationship")
            
            # Convexity benefit calculation
            convexity_adj = 0.5 * convexity * (0.01**2) * 100  # 1% yield change
            st.info(f"**Convexity Benefit**: Adds approximately {convexity_adj:.2f}% to price change estimates for large moves")
    
    # Duration Comparison Chart
    st.markdown("#### Duration Across Different Bond Characteristics")
    
    # Create duration comparison data
    maturities = [1, 3, 5, 10, 15, 20, 30]
    coupon_rates = [0.02, 0.04, 0.06]  # 2%, 4%, 6%
    yield_level = 0.05  # 5% yield
    
    fig_duration = go.Figure()
    
    colors = ['blue', 'green', 'red']
    for i, coup_rate in enumerate(coupon_rates):
        durations = []
        for maturity in maturities:
            dur = bond_duration(1000, coup_rate, yield_level, maturity)
            durations.append(dur)
        
        fig_duration.add_trace(go.Scatter(
            x=maturities, y=durations,
            mode='lines+markers',
            name=f'{coup_rate:.0%} Coupon',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6)
        ))
    
    fig_duration.update_layout(
        title="Duration vs Maturity for Different Coupon Rates",
        xaxis_title="Years to Maturity",
        yaxis_title="Duration (Years)",
        height=400
    )
    
    st.plotly_chart(fig_duration, use_container_width=True)
    
    with st.expander("📖 Understanding Duration and Convexity"):
        st.markdown("""
        ### Duration: The Foundation of Interest Rate Risk
        
        **Macaulay Duration** measures the weighted average time until cash flows are received:
        
        ```
        Duration = Σ[t × PV(CFt)] / Bond Price
        ```
        
        **Modified Duration** measures price sensitivity to yield changes:
        
        ```
        Modified Duration = Macaulay Duration / (1 + YTM)
        ```
        
        **Price Change Approximation:**
        
        ```
        ΔPrice/Price ≈ -Modified Duration × ΔYield
        ```
        
        ### Key Duration Insights:
        
        1. **Maturity Effect**: Longer maturity → Higher duration
        2. **Coupon Effect**: Lower coupon → Higher duration  
        3. **Yield Effect**: Lower yields → Higher duration
        4. **Zero-Coupon Bonds**: Duration equals maturity
        
        ### Convexity: Refining the Estimate
        
        Duration assumes a linear relationship, but the actual relationship is curved (convex).
        
        **Better Price Change Estimate:**
        
        ```
        ΔPrice/Price ≈ -Modified Duration × ΔYield + 0.5 × Convexity × (ΔYield)²
        ```
        
        ### Why Convexity Matters:
        
        - **Positive Convexity**: Bond prices rise more when yields fall than they decline when yields rise
        - **Large Moves**: Duration alone underestimates price changes for significant yield movements
        - **Portfolio Benefits**: Higher convexity securities provide better risk-return profiles
        - **Negative Convexity**: Some bonds (callable, prepayable) can exhibit negative convexity
        """)
    
    # === SECTION 4: YIELD CURVE ANALYSIS ===
    st.subheader("📈 Yield Curves: Term Structure of Interest Rates")
    
    st.markdown("""
    The **yield curve** plots interest rates across different maturities for securities of similar credit quality. 
    It provides critical information about market expectations for economic growth, inflation, and monetary policy.
    """)
    
    # Yield Curve Visualization
    maturities, normal_curve, inverted_curve, flat_curve = create_yield_curve_data()
    
    fig_yield_curves = go.Figure()
    
    fig_yield_curves.add_trace(go.Scatter(
        x=maturities, y=normal_curve*100,
        mode='lines+markers',
        name='Normal (Upward Sloping)',
        line=dict(color='green', width=3),
        marker=dict(size=6)
    ))
    
    fig_yield_curves.add_trace(go.Scatter(
        x=maturities, y=inverted_curve*100,
        mode='lines+markers', 
        name='Inverted (Downward Sloping)',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    fig_yield_curves.add_trace(go.Scatter(
        x=maturities, y=flat_curve*100,
        mode='lines+markers',
        name='Flat',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    fig_yield_curves.update_layout(
        title="Different Yield Curve Shapes and Their Implications",
        xaxis_title="Years to Maturity",
        yaxis_title="Yield (%)",
        height=500
    )
    
    st.plotly_chart(fig_yield_curves, use_container_width=True)
    
    # Real-World Yield Curve Data
    st.markdown("#### Current Yield Environment")
    
    try:
        # Fetch current Treasury rates
        tickers = {
            "^IRX": "3M Treasury",
            "^FVX": "5Y Treasury", 
            "^TNX": "10Y Treasury",
            "^TYX": "30Y Treasury"
        }
        
        current_yields = {}
        for ticker, name in tickers.items():
            try:
                data = yf.Ticker(ticker).history(period="5d")
                if not data.empty:
                    current_yields[name] = data['Close'].iloc[-1]
            except:
                continue
        
        if current_yields:
            yield_col1, yield_col2, yield_col3, yield_col4 = st.columns(4)
            cols = [yield_col1, yield_col2, yield_col3, yield_col4]
            
            for i, (name, yield_val) in enumerate(current_yields.items()):
                if i < len(cols):
                    cols[i].metric(name, f"{yield_val:.2f}%")
    except:
        st.info("Live yield data temporarily unavailable")
    
    with st.expander("📖 Understanding Yield Curve Shapes"):
        st.markdown("""
        ### Yield Curve Shapes and Economic Signals
        
        **1. Normal (Upward Sloping) Curve**
        - **Shape**: Short rates < Long rates
        - **Economic Signal**: Healthy growth expectations, normal risk premiums
        - **Monetary Policy**: Accommodative or neutral stance
        - **Investor Implication**: Steepness rewards extending duration
        
        **2. Inverted (Downward Sloping) Curve**  
        - **Shape**: Short rates > Long rates
        - **Economic Signal**: Recession expectations, aggressive tightening
        - **Monetary Policy**: Restrictive stance to combat inflation
        - **Historical Note**: Reliable recession predictor (6-18 months ahead)
        
        **3. Flat Curve**
        - **Shape**: Similar rates across maturities
        - **Economic Signal**: Uncertainty, transition period
        - **Monetary Policy**: End of hiking cycle or beginning of easing
        - **Trade Opportunities**: Limited duration premium
        
        ### Key Curve Relationships:
        
        **Steepening vs Flattening:**
        - **Steepening**: Long yields rise faster than short yields
        - **Flattening**: Short yields rise faster or long yields fall faster
        - **Bull Steepening**: Both decline, but short rates fall more
        - **Bear Flattening**: Both rise, but long rates rise less
        
        ### Practical Applications:
        
        - **Duration Positioning**: Steep curves favor longer duration
        - **Relative Value**: Compare securities across the curve
        - **Economic Forecasting**: Curve shape signals policy expectations
        - **Hedging Strategies**: Match asset-liability durations
        """)
    
    # === SECTION 5: CREDIT RISK AND SPREADS ===
    st.subheader("💳 Credit Risk: Beyond Risk-Free Rates")
    
    st.markdown("""
    **Credit spreads** represent the additional yield investors require to compensate for credit risk - 
    the possibility that an issuer may default on their obligations or experience a downgrade in creditworthiness.
    """)
    
    # Credit Spread Analysis
    dates, ig_spreads, hy_spreads = calculate_credit_spread_history()
    
    fig_credit_spreads = go.Figure()
    
    fig_credit_spreads.add_trace(go.Scatter(
        x=dates, y=ig_spreads,
        mode='lines',
        name='Investment Grade (IG)',
        line=dict(color='blue', width=2)
    ))
    
    fig_credit_spreads.add_trace(go.Scatter(
        x=dates, y=hy_spreads,
        mode='lines',
        name='High Yield (HY)',
        line=dict(color='red', width=2)
    ))
    
    fig_credit_spreads.update_layout(
        title="Credit Spreads Over Time (Basis Points Over Treasuries)",
        xaxis_title="Date",
        yaxis_title="Spread (Basis Points)",
        height=400
    )
    
    st.plotly_chart(fig_credit_spreads, use_container_width=True)
    
    # Credit Rating Overview
    st.markdown("#### Credit Rating Scale and Typical Spreads")
    
    rating_data = pd.DataFrame({
        'Rating': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],
        'Grade': ['Investment', 'Investment', 'Investment', 'Investment', 'High Yield', 'High Yield', 'High Yield'],
        'Typical Spread (bps)': [20, 40, 80, 150, 300, 500, 800],
        'Default Risk': ['Minimal', 'Very Low', 'Low', 'Low-Moderate', 'Moderate', 'High', 'Very High'],
        'Description': [
            'Highest credit quality',
            'Very strong capacity',
            'Strong capacity', 
            'Adequate capacity',
            'Speculative elements',
            'Significant credit risk',
            'Substantial risk'
        ]
    })
    
    st.dataframe(rating_data, use_container_width=True)
    
    with st.expander("📖 Understanding Credit Risk"):
        st.markdown("""
        ### Components of Credit Risk
        
        **1. Default Risk**
        - Probability that issuer fails to make payments
        - Ultimate credit event with potential total loss
        - Varies by issuer financial strength and economic cycle
        
        **2. Credit Migration Risk**
        - Risk of rating downgrades/upgrades
        - Can cause mark-to-market losses even without default
        - Particularly important for bonds near rating boundaries
        
        **3. Credit Spread Risk**
        - Market-driven changes in spread requirements
        - Reflects changing risk perception and market conditions
        - Can cause significant price volatility
        
        **4. Recovery Risk**
        - Uncertainty about recovery value in default
        - Depends on seniority, collateral, and market conditions
        - Typically 20-60% for corporate bonds
        
        ### Credit Spread Drivers:
        
        **Fundamental Factors:**
        - Issuer financial health and business prospects
        - Industry conditions and competitive position
        - Macroeconomic environment and credit cycle
        
        **Technical Factors:**
        - Supply and demand dynamics
        - Liquidity conditions and market structure
        - Risk appetite and investor sentiment
        
        ### Credit Spread Behavior:
        
        **Economic Expansion**: Spreads typically tighten as default risk decreases
        **Economic Contraction**: Spreads widen due to higher default expectations
        **Flight to Quality**: Crisis periods see dramatic spread widening
        **Recovery Phases**: Gradual spread normalization as conditions improve
        """)
    
    # === SECTION 6: FIXED INCOME STRATEGY FRAMEWORK ===
    st.subheader("🎯 Fixed Income in Portfolio Strategy")
    
    strategy_tabs = st.tabs(["📊 Risk-Return Profile", "🛡️ Portfolio Role", "⚙️ Implementation"])
    
    with strategy_tabs[0]:
        st.markdown("### Risk-Return Characteristics Across Fixed Income")
        
        # Risk-return scatter plot
        securities = [
            ('3M Treasury Bills', 0.5, 3.0),
            ('2Y Treasury Notes', 2.0, 4.0),
            ('10Y Treasury Bonds', 8.0, 4.5),
            ('30Y Treasury Bonds', 15.0, 5.0),
            ('IG Corporate 5Y', 5.0, 5.5),
            ('IG Corporate 10Y', 9.0, 6.0),
            ('High Yield 5Y', 12.0, 8.0),
            ('Emerging Market Debt', 15.0, 9.0),
            ('TIPS 10Y', 8.5, 4.8),
            ('Municipal Bonds 10Y', 8.0, 4.2)
        ]
        
        fig_risk_return = go.Figure()
        
        for name, risk, return_val in securities:
            color = 'green' if 'Treasury' in name else 'blue' if 'IG' in name else 'red' if 'High Yield' in name else 'orange'
            fig_risk_return.add_trace(go.Scatter(
                x=[risk], y=[return_val],
                mode='markers+text',
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=10, color=color),
                showlegend=False
            ))
        
        fig_risk_return.update_layout(
            title="Risk-Return Profile of Fixed Income Securities",
            xaxis_title="Duration Risk (Years)",
            yaxis_title="Expected Yield (%)",
            height=500
        )
        
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        st.markdown("""
        **Key Observations:**
        - **Risk-Free Rate**: Treasury securities provide duration risk without credit risk
        - **Credit Premium**: Corporate bonds offer higher yields for credit risk
        - **Maturity Premium**: Longer maturities typically provide higher yields
        - **Quality Spectrum**: Clear risk-return trade-off from Treasuries to High Yield
        """)
    
    with strategy_tabs[1]:
        st.markdown("### Strategic Roles in Portfolio Construction")
        
        roles_data = pd.DataFrame({
            'Role': ['Income Generation', 'Capital Preservation', 'Diversification', 'Liability Matching', 'Inflation Protection'],
            'Instruments': [
                'High-yield bonds, Dividend-focused funds',
                'Short-term Treasuries, High-grade corporates', 
                'Long-term Treasuries, Investment-grade bonds',
                'Duration-matched bonds, STRIPS',
                'TIPS, I-Bonds, Floating rate notes'
            ],
            'Key Consideration': [
                'Yield level vs credit risk',
                'Duration and credit quality',
                'Correlation with other assets',
                'Cash flow timing alignment', 
                'Real vs nominal returns'
            ]
        })
        
        st.dataframe(roles_data, use_container_width=True)
        
        st.markdown("""
        ### Portfolio Allocation Considerations:
        
        **Conservative Investors (60%+ Fixed Income):**
        - Focus on capital preservation and income
        - Emphasize high-quality, shorter-duration securities
        - Ladder maturities to reduce reinvestment risk
        
        **Balanced Investors (30-50% Fixed Income):**
        - Balance growth and stability objectives
        - Mix of government and corporate securities
        - Tactical duration and credit adjustments
        
        **Growth Investors (10-30% Fixed Income):**
        - Emphasize diversification benefits
        - Focus on liquid, high-quality securities
        - Consider inflation-protected securities
        """)
    
    with strategy_tabs[2]:
        st.markdown("### Implementation Approaches")
        
        implementation_col1, implementation_col2 = st.columns(2)
        
        with implementation_col1:
            st.markdown("**Active vs Passive Management**")
            
            active_passive_data = pd.DataFrame({
                'Approach': ['Index Funds/ETFs', 'Active Management', 'Core-Satellite', 'Laddered Portfolios'],
                'Best For': [
                    'Cost-conscious, broad exposure',
                    'Value-added strategies, credit selection', 
                    'Combines passive core with active tilts',
                    'Predictable cash flows, immunization'
                ],
                'Key Benefits': [
                    'Low fees, diversification',
                    'Potential for alpha generation',
                    'Balanced cost/performance approach', 
                    'Eliminates reinvestment risk'
                ]
            })
            
            st.dataframe(active_passive_data, use_container_width=True)
        
        with implementation_col2:
            st.markdown("**Duration and Credit Strategy**")
            
            strategy_scenarios = pd.DataFrame({
                'Market View': ['Rising Rates', 'Falling Rates', 'Stable Rates', 'Credit Stress'],
                'Duration Strategy': ['Shorten', 'Extend', 'Neutral', 'Quality Focus'],
                'Credit Strategy': ['Move up quality', 'Selective risk-taking', 'Balanced approach', 'Flight to quality'],
                'Expected Outcome': [
                    'Lower duration risk',
                    'Capital gains from rates',
                    'Steady carry income',
                    'Preserve capital'
                ]
            })
            
            st.dataframe(strategy_scenarios, use_container_width=True)
    
    # === SECTION 7: INTEREST RATE SCENARIOS ===
    st.subheader("📊 Interest Rate Scenarios and Portfolio Impact")
    
    st.markdown("""
    Understanding how different interest rate environments affect fixed income portfolios is crucial 
    for strategic asset allocation and risk management.
    """)
    
    # Interest Rate Scenario Analysis
    scenarios = simulate_interest_rate_scenarios()
    
    fig_scenarios = go.Figure()
    
    colors = {'Gradual Rise': 'blue', 'Shock Rise': 'red', 'Volatile Sideways': 'green'}
    
    for scenario_name, rates in scenarios.items():
        months = list(range(len(rates)))
        fig_scenarios.add_trace(go.Scatter(
            x=months, y=np.array(rates)*100,
            mode='lines',
            name=scenario_name,
            line=dict(color=colors[scenario_name], width=2)
        ))
    
    fig_scenarios.update_layout(
        title="Interest Rate Scenarios: 10-Year Projection",
        xaxis_title="Months",
        yaxis_title="Interest Rate (%)",
        height=400
    )
    
    st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Portfolio Impact Analysis
    st.markdown("#### Portfolio Impact Under Different Scenarios")
    
    # Calculate impact for sample bond
    sample_duration = 7.0  # 7-year duration bond
    
    scenario_impact = []
    for scenario_name, rates in scenarios.items():
        initial_rate = rates[0]
        final_rate = rates[-1]
        rate_change = final_rate - initial_rate
        
        # Duration-based price change
        price_change = -sample_duration * rate_change * 100  # Convert to percentage
        
        scenario_impact.append({
            'Scenario': scenario_name,
            'Rate Change': f"{rate_change*100:+.1f} bps",
            'Bond Price Impact': f"{price_change:+.1f}%",
            'Portfolio Implication': 'Negative' if price_change < -2 else 'Positive' if price_change > 2 else 'Neutral'
        })
    
    impact_df = pd.DataFrame(scenario_impact)
    st.dataframe(impact_df, use_container_width=True)
    
    # === SECTION 8: EDUCATIONAL SUMMARY ===
    st.subheader("📚 Key Learning Summary")
    
    with st.expander("🎯 Essential Fixed Income Concepts", expanded=True):
        st.markdown("""
        ### Core Principles to Remember:
        
        **1. Price-Yield Inverse Relationship**
        - Bond prices and yields move in opposite directions
        - This relationship is convex, not linear
        - Higher duration bonds show greater price sensitivity
        
        **2. Duration as Risk Measure**
        - Duration measures interest rate sensitivity
        - Modified duration gives approximate price change for yield changes
        - Convexity improves estimates for larger moves
        
        **3. Yield Curve Information**
        - Normal curve: Growth expectations, normal risk premiums
        - Inverted curve: Recession signal, restrictive monetary policy  
        - Flat curve: Uncertainty, policy transition periods
        
        **4. Credit Risk Components**
        - Default risk: Ultimate credit event with potential loss
        - Spread risk: Market-driven changes in credit premiums
        - Migration risk: Rating changes affecting valuations
        
        **5. Portfolio Role Versatility**
        - Income generation through coupon payments
        - Capital preservation via high-quality securities
        - Diversification benefits versus equity holdings
        - Liability matching through duration alignment
        
        ### Practical Applications:
        
        - **Interest Rate Views**: Adjust duration based on rate expectations
        - **Credit Cycle Management**: Move up/down credit spectrum based on economic cycle
        - **Yield Curve Strategies**: Position across curve based on shape expectations  
        - **Inflation Protection**: Use TIPS and floating rate notes for real return preservation
        """)
    
    with st.expander("🔧 Tools and Formulas"):
        st.markdown("""
        ### Key Formulas:
        
        **Bond Pricing:**
        ```
        Price = Σ[Coupon/(1+YTM)^t] + Face Value/(1+YTM)^n
        ```
        
        **Modified Duration:**
        ```
        Modified Duration = Macaulay Duration / (1 + YTM)
        ```
        
        **Price Change Approximation:**
        ```
        % Price Change ≈ -Modified Duration × Yield Change
        ```
        
        **With Convexity:**
        ```
        % Price Change ≈ -Duration × ΔY + 0.5 × Convexity × (ΔY)²
        ```
        
        **Current Yield:**
        ```
        Current Yield = Annual Coupon Payment / Current Market Price
        ```
        
        **Credit Spread:**
        ```
        Credit Spread = Corporate Yield - Government Yield (same maturity)
        ```
        
        ### Quick Reference:
        
        - **Basis Point**: 0.01% or 1/100th of a percent
        - **Par Value**: Typically $1,000 for corporate bonds
        - **Premium**: Bond trading above par value
        - **Discount**: Bond trading below par value
        - **Yield to Maturity**: Total return if held to maturity
        """)
    
    # === FOOTER ===
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        <p><strong>Fixed Income Fundamentals</strong></p>
        <p>Educational framework for understanding bonds, yields, and interest rate risk</p>
        <p><em>For educational purposes only - Not investment advice</em></p>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# Run the page
if __name__ == "__main__":
    show_fixed_income_page()
