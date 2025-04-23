import streamlit as st
import numpy as np
import math
from math import log, sqrt, exp, pi
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Related Formula: Black-Scholes
# -----------------------------
def black_scholes_price(S, K, T, r, q, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    return price

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

def total_strategy_metrics(S, T, r, q, legs):
    total_premium = total_delta = total_gamma = total_vega = total_theta = total_rho = 0.0
    for leg in legs:
        K = leg['strike']
        sigma = leg['vol']
        n = leg['notional']
        otype = leg['type']
        price_leg = black_scholes_price(S, K, T, r, q, sigma, otype)
        greeks = black_scholes_greeks(S, K, T, r, q, sigma, otype)
        total_premium += n * price_leg
        total_delta    += n * greeks['delta']
        total_gamma    += n * greeks['gamma']
        total_vega     += n * greeks['vega']
        total_theta    += n * greeks['theta']
        total_rho      += n * greeks['rho']
    return {
        'premium': total_premium,
        'delta':   total_delta,
        'gamma':   total_gamma,
        'vega':    total_vega,
        'theta':   total_theta,
        'rho':     total_rho
    }

def payoff_at_expiry(S_values, legs):
    payoffs = []
    for S_ in S_values:
        total = 0.0
        for leg in legs:
            K = leg['strike']
            n = leg['notional']
            otype = leg['type']
            if otype == 'call':
                total += n * max(S_ - K, 0.0)
            else:
                total += n * max(K - S_, 0.0)
        payoffs.append(total)
    return np.array(payoffs)

def strategy_metrics_curve(S_values, T, r, q, legs):
    premiums, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []
    for S_ in S_values:
        # To avoid math domain errors, replace S=0 with a small epsilon
        S_val = S_ if S_ > 0 else 1e-3
        res = total_strategy_metrics(S_val, T, r, q, legs)
        premiums.append(res['premium'])
        deltas.append(res['delta'])
        gammas.append(res['gamma'])
        vegas.append(res['vega'])
        thetas.append(res['theta'])
        rhos.append(res['rho'])
    return {
        'premium': np.array(premiums),
        'delta': np.array(deltas),
        'gamma': np.array(gammas),
        'vega': np.array(vegas),
        'theta': np.array(thetas),
        'rho': np.array(rhos),
    }

# ---------------------------
# Helper Function for Plotting (Single Figure)
# ---------------------------
def create_metric_fig(S_range, y_data, metric_name, color, legs, S_current):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(S_range, y_data, label=metric_name, color=color)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Underlying Price")
    for leg in legs:
        ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
    ax.axvline(S_current, color='black', linestyle=':', alpha=0.7, label='Current Spot')
    ax.legend()
    ax.set_xlim(0, 200)
    plt.tight_layout()
    return fig

# ------------------------------------------------
# Main Streamlit App Layout
# ------------------------------------------------
def main():
    st.set_page_config(layout="wide")  # optional, for a wide layout
    st.title("Option Strategies Pricer with Multiple Graphs")
    st.sidebar.title("Input Parameters")

    # Strategy selection and basic inputs
    strategy = st.sidebar.selectbox(
        "Select Option Strategy",
        ("Call Spread", "Put Spread", "Straddle", "Strangle", "Risk Reversal", "Butterfly Spread", "Condor Spread")
    )
    S = st.sidebar.number_input("Spot Price (S)", value=100.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.02)
    q = st.sidebar.number_input("Dividend Yield (q)", value=0.0)
    T = st.sidebar.number_input("Time to Maturity (years)", value=1.0)

    # Define strategy legs based on selection
    if strategy == "Call Spread":
        st.subheader("Call Spread Legs")
        strike_low = st.number_input("Lower Strike (K1)", value=90.0)
        vol_low    = st.number_input("Vol for K1", value=0.20)
        strike_high = st.number_input("Higher Strike (K2)", value=110.0)
        vol_high   = st.number_input("Vol for K2", value=0.20)
        legs = [
            {'type': 'call', 'strike': strike_low, 'vol': vol_low, 'notional': 1},
            {'type': 'call', 'strike': strike_high, 'vol': vol_high, 'notional': -1},
        ]
    elif strategy == "Put Spread":
        st.subheader("Put Spread Legs")
        strike_low = st.number_input("Lower Strike (K1)", value=90.0)
        vol_low    = st.number_input("Vol for K1", value=0.20)
        strike_high = st.number_input("Higher Strike (K2)", value=110.0)
        vol_high   = st.number_input("Vol for K2", value=0.20)
        legs = [
            {'type': 'put', 'strike': strike_low, 'vol': vol_low, 'notional': -1},
            {'type': 'put', 'strike': strike_high, 'vol': vol_high, 'notional': 1},
        ]
    elif strategy == "Straddle":
        st.subheader("Straddle Legs")
        strike = st.number_input("Strike (K)", value=100.0)
        vol_call = st.number_input("Vol for Call", value=0.20)
        vol_put  = st.number_input("Vol for Put", value=0.20)
        legs = [
            {'type': 'call', 'strike': strike, 'vol': vol_call, 'notional': 1},
            {'type': 'put', 'strike': strike, 'vol': vol_put, 'notional': 1},
        ]
    elif strategy == "Strangle":
        st.subheader("Strangle Legs")
        strike_put  = st.number_input("Put Strike (K1)", value=90.0)
        vol_put     = st.number_input("Put Vol", value=0.20)
        strike_call = st.number_input("Call Strike (K2)", value=110.0)
        vol_call    = st.number_input("Call Vol", value=0.20)
        legs = [
            {'type': 'put', 'strike': strike_put, 'vol': vol_put, 'notional': 1},
            {'type': 'call', 'strike': strike_call, 'vol': vol_call, 'notional': 1},
        ]
    elif strategy == "Risk Reversal":
        st.subheader("Risk Reversal Legs")
        strike_put  = st.number_input("Put Strike (K Put)", value=95.0)
        vol_put     = st.number_input("Put Vol", value=0.20)
        strike_call = st.number_input("Call Strike (K Call)", value=105.0)
        vol_call    = st.number_input("Call Vol", value=0.20)
        legs = [
            {'type': 'put', 'strike': strike_put, 'vol': vol_put, 'notional': -1},
            {'type': 'call', 'strike': strike_call, 'vol': vol_call, 'notional': 1},
        ]
    elif strategy == "Butterfly Spread":
        st.subheader("Butterfly Spread Legs")
        K1 = st.number_input("K1 (Low)", value=90.0)
        vol1 = st.number_input("Vol K1", value=0.20)
        K2 = st.number_input("K2 (Middle)", value=100.0)
        vol2 = st.number_input("Vol K2", value=0.20)
        K3 = st.number_input("K3 (High)", value=110.0)
        vol3 = st.number_input("Vol K3", value=0.20)
        legs = [
            {'type': 'call', 'strike': K1, 'vol': vol1, 'notional': 1},
            {'type': 'call', 'strike': K2, 'vol': vol2, 'notional': -2},
            {'type': 'call', 'strike': K3, 'vol': vol3, 'notional': 1},
        ]
    elif strategy == "Condor Spread":
        st.subheader("Condor Spread Legs")
        K1 = st.number_input("K1 (Lowest)", value=80.0)
        vol1 = st.number_input("Vol K1", value=0.20)
        K2 = st.number_input("K2 (Low)", value=90.0)
        vol2 = st.number_input("Vol K2", value=0.20)
        K3 = st.number_input("K3 (High)", value=110.0)
        vol3 = st.number_input("Vol K3", value=0.20)
        K4 = st.number_input("K4 (Highest)", value=120.0)
        vol4 = st.number_input("Vol K4", value=0.20)
        legs = [
            {'type': 'call', 'strike': K1, 'vol': vol1, 'notional': 1},
            {'type': 'call', 'strike': K2, 'vol': vol2, 'notional': -1},
            {'type': 'call', 'strike': K3, 'vol': vol3, 'notional': -1},
            {'type': 'call', 'strike': K4, 'vol': vol4, 'notional': 1},
        ]

    # --- Calculate current strategy metrics at the given spot price ---
    current_results = total_strategy_metrics(S, T, r, q, legs)
    st.markdown("### Current Strategy Value (at Spot = {:.2f})".format(S))
    st.write(f"- **Premium:** {current_results['premium']:.4f}")
    st.write(f"- **Delta:**   {current_results['delta']:.4f}")
    st.write(f"- **Gamma:**   {current_results['gamma']:.6f}")
    st.write(f"- **Vega:**    {current_results['vega']:.4f}")
    st.write(f"- **Theta:**   {current_results['theta']:.4f}")
    st.write(f"- **Rho:**     {current_results['rho']:.4f}")

    # --- Prepare range of underlying prices to visualize ---
    S_min = max(1.0, S * 0.1)
    S_max = S * 2.2
    n_points = 100
    S_range = np.linspace(S_min, S_max, n_points)

    # --- Compute metrics across S_range ---
    payoff_vals = payoff_at_expiry(S_range, legs)
    curves = strategy_metrics_curve(S_range, T, r, q, legs)

    # --- Create Tabs for Individual Graphs ---
    tab_names = ["Payoff", "Premium", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    tabs = st.tabs(tab_names)

    # Tab 1: Payoff
    with tabs[0]:
        st.subheader("Payoff at Expiry")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, payoff_vals, label='Payoff', color='blue')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price at Expiry")
        ax.set_ylabel("Payoff")
        ax.set_title(f"{strategy} Payoff at Expiry")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 2: Premium
    with tabs[1]:
        st.subheader("Strategy Premium vs. Underlying Price (Today)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['premium'], label='Premium', color='green')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Premium")
        ax.set_title(f"{strategy} - Premium")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 3: Delta
    with tabs[2]:
        st.subheader("Delta vs. Underlying Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['delta'], label='Delta', color='blue')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Delta")
        ax.set_title(f"{strategy} - Delta")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 4: Gamma
    with tabs[3]:
        st.subheader("Gamma vs. Underlying Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['gamma'], label='Gamma', color='orange')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Gamma")
        ax.set_title(f"{strategy} - Gamma")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 5: Vega
    with tabs[4]:
        st.subheader("Vega vs. Underlying Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['vega'], label='Vega', color='purple')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Vega")
        ax.set_title(f"{strategy} - Vega")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 6: Theta
    with tabs[5]:
        st.subheader("Theta vs. Underlying Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['theta'], label='Theta', color='brown')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Theta")
        ax.set_title(f"{strategy} - Theta")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 7: Rho
    with tabs[6]:
        st.subheader("Rho vs. Underlying Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(S_range, curves['rho'], label='Rho', color='teal')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price (Now)")
        ax.set_ylabel("Rho")
        ax.set_title(f"{strategy} - Rho")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------
    # Combined Grid of Six Graphs (Premium and Greeks) in One Figure
    # ---------------------------
    st.markdown("## Combined Greek & Premium Grid")
    metric_info = {
        "Premium": {"data": curves['premium'], "color": "green"},
        "Delta":   {"data": curves['delta'],   "color": "blue"},
        "Gamma":   {"data": curves['gamma'],   "color": "orange"},
        "Vega":    {"data": curves['vega'],    "color": "purple"},
        "Theta":   {"data": curves['theta'],   "color": "brown"},
        "Rho":     {"data": curves['rho'],     "color": "teal"},
    }
    metrics = ["Premium", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        ax.plot(S_range, metric_info[metric]["data"], label=metric, color=metric_info[metric]["color"])
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("Underlying Price")
        ax.set_ylabel(metric)
        ax.set_title(f"{strategy} - {metric}")
        for leg in legs:
            ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
        ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
        ax.legend(fontsize='small')
        ax.set_xlim(0, 200)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
