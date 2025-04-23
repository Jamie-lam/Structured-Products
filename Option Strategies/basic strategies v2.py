import streamlit as st
import numpy as np
import math
from math import log, sqrt, exp, pi
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# --------------------------------------------------
# Binomial tree for American or Barrier options
# --------------------------------------------------

def binomial_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    steps: int = 200,
    option_type: str = 'call',
    american: bool = True,
    barrier: float = None,
    barrier_type: str = 'up-and-out',
) -> float:
    """
    General binomial pricer.

    Parameters
    ----------
    S : float
        Spot price today.
    K : float
        Option strike.
    T : float
        Time to maturity (in years).
    r : float
        Continuously‐compounded risk‐free rate.
    q : float
        Continuously‐compounded dividend yield.
    sigma : float
        Volatility (annual).
    steps : int, default 200
        Number of binomial steps.
    option_type : {'call', 'put'}, default 'call'
        Option payoff style.
    american : bool, default True
        If True, allows early exercise (American style).
    barrier : float or None
        Barrier level for knock‐out; if None, vanilla option.
    barrier_type : {'up-and-out', 'down-and-out'}, default 'up-and-out'
        Knock‐out barrier style.

    Returns
    -------
    float
        Present value of the option.
    """
    # 1️⃣ Set up parameters
    dt   = T / steps
    u    = math.exp(sigma * math.sqrt(dt))
    d    = 1 / u
    disc = math.exp(-r * dt)
    p    = (math.exp((r - q) * dt) - d) / (u - d)

    # 2️⃣ Stock‐price lattice at maturity
    ST = np.array([S * u**j * d**(steps - j) for j in range(steps + 1)])

    # 3️⃣ Initial payoff at maturity (with barrier if any)
    if barrier is not None:
        if barrier_type == 'up-and-out':
            alive = ST < barrier
        elif barrier_type == 'down-and-out':
            alive = ST > barrier
        else:
            raise ValueError("Unsupported barrier_type")
    else:
        alive = np.ones_like(ST, dtype=bool)

    if option_type == 'call':
        option_values = np.where(alive, np.maximum(ST - K, 0.0), 0.0)
    else:
        option_values = np.where(alive, np.maximum(K - ST, 0.0), 0.0)

    # 4️⃣ Backward induction
    for step in range(steps, 0, -1):
        # → Move to previous node: align via upper slice
        ST = ST[1:] / u           # now length == step

        # → Discounted expectation
        option_values = disc * (
            p * option_values[1:] + (1 - p) * option_values[:-1]
        )

        # → Apply barrier knock‐out
        if barrier is not None:
            if barrier_type == 'up-and-out':
                alive = ST < barrier
            else:
                alive = ST > barrier
            option_values = np.where(alive, option_values, 0.0)

        # → Early exercise for American style
        if american:
            intrinsic = (
                np.maximum(ST - K, 0.0)
                if option_type == 'call'
                else np.maximum(K - ST, 0.0)
            )
            option_values = np.maximum(option_values, intrinsic)

    return float(option_values[0])

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

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Input Parameters")
    mode = st.sidebar.radio("Select Mode", ("Single Vanilla Option", "Option Strategies"))

    # Common inputs
    S     = st.sidebar.number_input("Spot Price (S)", value=100.0, format="%f")
    r     = st.sidebar.number_input("Risk-Free Rate (r)", value=0.02, format="%f")
    q     = st.sidebar.number_input("Dividend Yield (q)", value=0.0, format="%f")
    T     = st.sidebar.number_input("Time to Maturity (years)", value=1.0, format="%f")
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, format="%f")

    if mode == "Single Vanilla Option":
        st.title("Vanilla / Barrier Option Pricer")
        option_type     = st.selectbox("Option Type", ("call", "put"))
        style           = st.selectbox("Exercise Style", ("European", "American"))
        barrier_enabled = st.checkbox("Barrier Option")

        barrier = None
        barrier_type = None
        if barrier_enabled:
            barrier_type = st.selectbox("Barrier Type", ("up-and-out", "down-and-out"))
            barrier = st.number_input(
                "Barrier Level (B)",
                value=120.0 if barrier_type == "up-and-out" else 80.0
            )
            if style == "European":
                st.info(
                    "Barrier pricing uses a **binomial approximation** for both European "
                    "and American cases. You can swap in analytic formulas if you like."
                )

        K     = st.number_input("Strike (K)", value=100.0, format="%f")
        steps = st.slider("Binomial Steps", 50, 1000, 400)

        if st.button("Calculate Price"):
            if style == "European" and not barrier_enabled:
                price = black_scholes_price(S, K, T, r, q, sigma, option_type)
            else:
                price = binomial_option_price(
                    S, K, T, r, q, sigma,
                    steps=steps,
                    option_type=option_type,
                    american=(style == "American"),
                    barrier=barrier,
                    barrier_type=barrier_type,
                )
            st.success(f"Calculated {style} {option_type.capitalize()} Price: {price:.4f}")

    else:
        st.title("Option Strategies Pricer with Multiple Graphs")
        strategy = st.sidebar.selectbox(
            "Select Option Strategy",
            (
                "Call Spread", "Put Spread", "Straddle", "Strangle",
                "Risk Reversal", "Butterfly Spread", "Condor Spread",
            ),
        )

        # Build legs
        if strategy == "Call Spread":
            st.subheader("Call Spread Legs")
            K1  = st.number_input("Lower Strike (K1)", value=90.0)
            v1  = st.number_input("Vol for K1",     value=0.20)
            K2  = st.number_input("Higher Strike (K2)", value=110.0)
            v2  = st.number_input("Vol for K2",     value=0.20)
            legs = [
                {'type':'call','strike':K1,'vol':v1,'notional':1},
                {'type':'call','strike':K2,'vol':v2,'notional':-1},
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

        # Calculate current metrics
        results = total_strategy_metrics(S, T, r, q, legs)
        st.markdown(f"### Current Strategy Metrics @ Spot = {S:.2f}")
        st.write(f"- **Premium:** {results['premium']:.4f}")
        st.write(f"- **Delta:**   {results['delta']:.4f}")
        st.write(f"- **Gamma:**   {results['gamma']:.6f}")
        st.write(f"- **Vega:**    {results['vega']:.4f}")
        st.write(f"- **Theta:**   {results['theta']:.4f}")
        st.write(f"- **Rho:**     {results['rho']:.4f}")

        # Prepare underlying range
        S_min    = max(1.0, S * 0.1)
        S_max    = S * 2.2
        S_range  = np.linspace(S_min, S_max, 100)
        payoff   = payoff_at_expiry(S_range, legs)
        curves   = strategy_metrics_curve(S_range, T, r, q, legs)

        # Tabs for individual graphs
        tab_names = ["Payoff","Premium","Delta","Gamma","Vega","Theta","Rho"]
        tabs = st.tabs(tab_names)

        colors = {
            "Payoff":"blue", "Premium":"green", "Delta":"blue",
            "Gamma":"orange","Vega":"purple","Theta":"brown","Rho":"teal"
        }

        for i, name in enumerate(tab_names):
            with tabs[i]:
                st.subheader(f"{name} vs Underlying Price")
                fig, ax = plt.subplots(figsize=(10,4))
                data = payoff if name=="Payoff" else curves[name.lower()]
                ax.plot(S_range, data, label=name, color=colors[name])
                ax.axhline(0, color='gray', linestyle='--')
                ax.set_xlabel("Underlying Price")
                ax.set_ylabel(name)
                ax.set_title(f"{strategy} – {name}")
                for leg in legs:
                    ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
                ax.axvline(S, color='black', linestyle=':', alpha=0.7, label='Current Spot')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

        # Combined 2x3 grid (omit payoff)
        st.markdown("## Combined Greek & Premium Grid")
        metrics = ["premium","delta","gamma","vega","theta","rho"]
        fig, axs = plt.subplots(2, 3, figsize=(15,8))
        for idx, metric in enumerate(metrics):
            row, col = divmod(idx, 3)
            ax = axs[row, col]
            ax.plot(S_range, curves[metric], label=metric.capitalize(), color=colors[metric.capitalize()])
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_title(f"{strategy} – {metric.capitalize()}")
            ax.set_xlabel("Underlying Price")
            ax.set_ylabel(metric.capitalize())
            for leg in legs:
                ax.axvline(leg['strike'], color='red', linestyle='--', alpha=0.4)
            ax.axvline(S, color='black', linestyle=':', alpha=0.7)
            ax.legend(fontsize='small')
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
