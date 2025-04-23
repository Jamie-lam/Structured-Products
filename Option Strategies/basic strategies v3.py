import streamlit as st
import numpy as np
import pandas as pd
import math
from math import log, sqrt, exp, pi
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Closed‑form Black–Scholes for European vanilla
# --------------------------------------------------
def black_scholes_price(S, K, T, r, q, sigma, option_type='call'):
    """European Black‑Scholes price for call or put."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# --------------------------------------------------
# Black–Scholes Greeks
# --------------------------------------------------
def black_scholes_greeks(S, K, T, r, q, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        delta = 1.0 if (option_type == 'call' and S > K) else (-1.0 if (option_type == 'put' and K > S) else 0.0)
        return {'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    if option_type == 'call':
        delta = exp(-q * T) * Nd1
        rho_factor = K * T * exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -exp(-q * T) * norm.cdf(-d1)
        rho_factor = -K * T * exp(-r * T) * norm.cdf(-d2) / 100
    gamma = exp(-q * T) * nd1 / (S * sigma * sqrt(T))
    vega  = S * sqrt(T) * exp(-q * T) * nd1 / 100
    if option_type == 'call':
        theta_annual = (- (S * nd1 * sigma * exp(-q * T)) / (2.0 * sqrt(T))
                        - r * K * exp(-r * T) * norm.cdf(d2)
                        + q * S * exp(-q * T) * Nd1)
    else:
        theta_annual = (- (S * nd1 * sigma * exp(-q * T)) / (2.0 * sqrt(T))
                        + r * K * exp(-r * T) * norm.cdf(-d2)
                        - q * S * exp(-q * T) * norm.cdf(-d1))
    theta = theta_annual / 256
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho_factor}

# --------------------------------------------------
# Strategy Metrics Helpers
# --------------------------------------------------
def total_strategy_metrics(S, T, r, q, legs):
    total_premium = total_delta = total_gamma = total_vega = total_theta = total_rho = 0.0
    for leg in legs:
        price = black_scholes_price(S, leg['strike'], T, r, q, leg['vol'], leg['type'])
        greeks = black_scholes_greeks(S, leg['strike'], T, r, q, leg['vol'], leg['type'])
        n = leg['notional']
        total_premium += n * price
        total_delta   += n * greeks['delta']
        total_gamma   += n * greeks['gamma']
        total_vega    += n * greeks['vega']
        total_theta   += n * greeks['theta']
        total_rho     += n * greeks['rho']
    return {
        'premium': total_premium,
        'delta':   total_delta,
        'gamma':   total_gamma,
        'vega':    total_vega,
        'theta':   total_theta,
        'rho':     total_rho
    }

def payoff_at_expiry(S_vals, legs):
    payoffs = []
    for S0 in S_vals:
        total = 0.0
        for leg in legs:
            if leg['type'] == 'call':
                total += leg['notional'] * max(S0 - leg['strike'], 0.0)
            else:
                total += leg['notional'] * max(leg['strike'] - S0, 0.0)
        payoffs.append(total)
    return np.array(payoffs)

def strategy_metrics_curve(S_vals, T, r, q, legs):
    arr = {'premium': [], 'delta': [], 'gamma': [], 'vega': [], 'theta': [], 'rho': []}
    for S0 in S_vals:
        res = total_strategy_metrics(S0 if S0>0 else 1e-3, T, r, q, legs)
        for k in arr:
            arr[k].append(res[k])
    return {k: np.array(v) for k, v in arr.items()}

# --------------------------------------------------
# Binomial Tree (American/Barrier)
# --------------------------------------------------
def binomial_option_price(
    S, K, T, r, q, sigma, steps=200,
    option_type='call', american=True,
    barrier=None, barrier_type='up-and-out'
):
    dt = T / steps
    u  = math.exp(sigma * math.sqrt(dt))
    d  = 1 / u
    disc = math.exp(-r * dt)
    p  = (math.exp((r - q) * dt) - d) / (u - d)
    ST = np.array([S * u**j * d**(steps-j) for j in range(steps+1)])
    alive = np.ones_like(ST, dtype=bool)
    if barrier is not None:
        alive = (ST < barrier) if barrier_type=='up-and-out' else (ST>barrier)
    if option_type=='call':
        vals = np.where(alive, np.maximum(ST-K,0),0)
    else:
        vals = np.where(alive, np.maximum(K-ST,0),0)
    for step in range(steps, 0, -1):
        ST = ST[1:]/u
        vals = disc*(p*vals[1:]+(1-p)*vals[:-1])
        if barrier is not None:
            alive = (ST<barrier) if barrier_type=='up-and-out' else (ST>barrier)
            vals = np.where(alive, vals, 0)
        if american:
            intrinsic = np.where(option_type=='call', np.maximum(ST-K,0), np.maximum(K-ST,0))
            vals = np.maximum(vals, intrinsic)
    return float(vals[0])

# --------------------------------------------------
# Interactive Plotly Helpers
# --------------------------------------------------
def create_interactive_metric_fig(S_range, y_data, metric_name, legs, S_current):
    df = pd.DataFrame({'Underlying': S_range, metric_name: y_data})
    fig = px.line(df, x='Underlying', y=metric_name,
                  title=f"{metric_name} vs Underlying Price",
                  labels={'Underlying':'Spot Price', metric_name:metric_name},
                  hover_data={metric_name:':.4f'})
    for leg in legs:
        fig.add_vline(x=leg['strike'], line=dict(color='red', dash='dash'),
                      annotation_text=f"K={leg['strike']}", annotation_position='top left', opacity=0.4)
    fig.add_vline(x=S_current, line=dict(color='black', dash='dot'),
                  annotation_text='Spot', annotation_position='top right', opacity=0.7)
    fig.update_layout(hovermode='x unified', margin=dict(l=40, r=20, t=50, b=40))
    return fig

def create_combined_grid(S_range, curves, legs, S_current):
    metrics = ['Premium','Delta','Gamma','Vega','Theta','Rho']
    fig = make_subplots(rows=2, cols=3, subplot_titles=metrics,
                        shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.15)
    for idx, m in enumerate(metrics):
        row, col = divmod(idx, 3)
        series = curves[m.lower()]
        fig.add_trace(go.Scatter(x=S_range, y=series, name=m,
                                 hovertemplate=f"{m}: %{{y:.4f}}<extra></extra>"),
                      row=row+1, col=col+1)
        for leg in legs:
            fig.add_vline(x=leg['strike'], line=dict(color='red', dash='dash'),
                          row=row+1, col=col+1)
        fig.add_vline(x=S_current, line=dict(color='black', dash='dot'),
                      row=row+1, col=col+1)
        fig.update_xaxes(title_text='Underlying Price', row=row+1, col=col+1)
        fig.update_yaxes(title_text=m, row=row+1, col=col+1)
    fig.update_layout(height=800, width=1200, hovermode='x unified',
                      showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return fig

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.set_page_config(layout='wide')
    st.sidebar.title('Input Parameters')
    mode = st.sidebar.radio('Select Mode', ('Single Vanilla Option','Option Strategies'))

    # Common inputs
    S     = st.sidebar.number_input('Spot Price (S)', value=100.0, format='%f')
    r     = st.sidebar.number_input('Risk-Free Rate (r)', value=0.02, format='%f')
    q     = st.sidebar.number_input('Dividend Yield (q)', value=0.0, format='%f')
    T     = st.sidebar.number_input('Time to Maturity (years)', value=1.0, format='%f')
    sigma = st.sidebar.number_input('Volatility (σ)', value=0.20, format='%f')

    if mode == 'Single Vanilla Option':
        st.title('Vanilla / Barrier Option Pricer')
        option_type     = st.selectbox('Option Type', ('call','put'))
        style           = st.selectbox('Exercise Style', ('European','American'))
        barrier_enabled = st.checkbox('Barrier Option')

        barrier = None
        barrier_type = None
        if barrier_enabled:
            barrier_type = st.selectbox('Barrier Type', ('up-and-out','down-and-out'))
            barrier = st.number_input('Barrier Level (B)',
                                       value=120.0 if barrier_type=='up-and-out' else 80.0)
            if style=='European':
                st.info('Barrier pricing uses a **binomial approximation** for both European and American cases.')

        K     = st.number_input('Strike (K)', value=100.0, format='%f')
        steps = st.slider('Binomial Steps', 50, 1000, 400)

        if st.button('Calculate Price'):
            if style=='European' and not barrier_enabled:
                price = black_scholes_price(S, K, T, r, q, sigma, option_type)
            else:
                price = binomial_option_price(S, K, T, r, q, sigma,
                                              steps=steps, option_type=option_type,
                                              american=(style=='American'),
                                              barrier=barrier, barrier_type=barrier_type)
            st.success(f'Calculated {style} {option_type.capitalize()} Price: {price:.4f}')

    else:
        st.title('Option Strategies Pricer with Interactive Plots')
        strategy = st.sidebar.selectbox('Select Option Strategy',
                                        ('Call Spread','Put Spread','Straddle','Strangle',
                                         'Risk Reversal','Butterfly Spread','Condor Spread'))
        # Initialize legs
        legs = []
        if strategy == 'Call Spread':
            st.subheader('Call Spread Legs')
            K1 = st.number_input('Lower Strike (K1)', value=90.0)
            v1 = st.number_input('Vol for K1', value=0.20)
            K2 = st.number_input('Higher Strike (K2)', value=110.0)
            v2 = st.number_input('Vol for K2', value=0.20)
            legs = [
                {'type':'call','strike':K1,'vol':v1,'notional':1},
                {'type':'call','strike':K2,'vol':v2,'notional':-1},
            ]
        elif strategy == 'Put Spread':
            st.subheader('Put Spread Legs')
            K1 = st.number_input('Lower Strike (K1)', value=90.0)
            v1 = st.number_input('Vol for K1', value=0.20)
            K2 = st.number_input('Higher Strike (K2)', value=110.0)
            v2 = st.number_input('Vol for K2', value=0.20)
            legs = [
                {'type':'put','strike':K1,'vol':v1,'notional':-1},
                {'type':'put','strike':K2,'vol':v2,'notional':1},
            ]
        elif strategy == 'Straddle':
            st.subheader('Straddle Legs')
            K0 = st.number_input('Strike (K)', value=100.0)
            vc = st.number_input('Vol for Call', value=0.20)
            vp = st.number_input('Vol for Put', value=0.20)
            legs = [
                {'type':'call','strike':K0,'vol':vc,'notional':1},
                {'type':'put','strike':K0,'vol':vp,'notional':1},
            ]
        elif strategy == 'Strangle':
            st.subheader('Strangle Legs')
            Kp = st.number_input('Put Strike (K1)', value=90.0)
            vp = st.number_input('Put Vol', value=0.20)
            Kc = st.number_input('Call Strike (K2)', value=110.0)
            vc = st.number_input('Call Vol', value=0.20)
            legs = [
                {'type':'put','strike':Kp,'vol':vp,'notional':1},
                {'type':'call','strike':Kc,'vol':vc,'notional':1},
            ]
        elif strategy == 'Risk Reversal':
            st.subheader('Risk Reversal Legs')
            Kp = st.number_input('Put Strike (K Put)', value=95.0)
            vp = st.number_input('Put Vol', value=0.20)
            Kc = st.number_input('Call Strike (K Call)', value=105.0)
            vc = st.number_input('Call Vol', value=0.20)
            legs = [
                {'type':'put','strike':Kp,'vol':vp,'notional':-1},
                {'type':'call','strike':Kc,'vol':vc,'notional':1},
            ]
        elif strategy == 'Butterfly Spread':
            st.subheader('Butterfly Spread Legs')
            K1 = st.number_input('K1 (Low)', value=90.0)
            v1 = st.number_input('Vol K1', value=0.20)
            K2 = st.number_input('K2 (Middle)', value=100.0)
            v2 = st.number_input('Vol K2', value=0.20)
            K3 = st.number_input('K3 (High)', value=110.0)
            v3 = st.number_input('Vol K3', value=0.20)
            legs = [
                {'type':'call','strike':K1,'vol':v1,'notional':1},
                {'type':'call','strike':K2,'vol':v2,'notional':-2},
                {'type':'call','strike':K3,'vol':v3,'notional':1},
            ]
        else:  # Condor Spread
            st.subheader('Condor Spread Legs')
            K1 = st.number_input('K1 (Lowest)', value=80.0)
            v1 = st.number_input('Vol K1', value=0.20)
            K2 = st.number_input('K2 (Low)', value=90.0)
            v2 = st.number_input('Vol K2', value=0.20)
            K3 = st.number_input('K3 (High)', value=110.0)
            v3 = st.number_input('Vol K3', value=0.20)
            K4 = st.number_input('K4 (Highest)', value=120.0)
            v4 = st.number_input('Vol K4', value=0.20)
            legs = [
                {'type':'call','strike':K1,'vol':v1,'notional':1},
                {'type':'call','strike':K2,'vol':v2,'notional':-1},
                {'type':'call','strike':K3,'vol':v3,'notional':-1},
                {'type':'call','strike':K4,'vol':v4,'notional':1},
            ]

        # Current metrics
        results = total_strategy_metrics(S, T, r, q, legs)
        st.markdown(f"### Current Strategy Metrics @ Spot = {S:.2f}")
        for key in ['premium','delta','gamma','vega','theta','rho']:
            st.write(f"- **{key.capitalize()}:** {results[key]:.4f}")

        # Prepare curves
        S_min, S_max = max(1.0, S*0.1), S*2.2
        S_range = np.linspace(S_min, S_max, 100)
        payoff = payoff_at_expiry(S_range, legs)
        curves = strategy_metrics_curve(S_range, T, r, q, legs)

        # Individual tabs
        tab_names = ['Payoff','Premium','Delta','Gamma','Vega','Theta','Rho']
        tabs = st.tabs(tab_names)
        for i, name in enumerate(tab_names):
            data = payoff if name=='Payoff' else curves[name.lower()]
            with tabs[i]:
                fig = create_interactive_metric_fig(S_range, data, name, legs, S)
                st.plotly_chart(fig, use_container_width=True)

        # Combined grid
        grid_fig = create_combined_grid(S_range, curves, legs, S)
        st.markdown('## Combined Greek & Premium Grid')
        st.plotly_chart(grid_fig, use_container_width=True)

if __name__ == '__main__':
    main()
