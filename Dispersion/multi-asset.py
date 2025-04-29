import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import Optional

# --------------------------------------------------
# Utility: Nearest Positive‑Definite (safety helper)
# --------------------------------------------------

def nearest_pd(corr: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Project a correlation matrix onto the PD cone via eigenvalue clipping."""
    A = (corr + corr.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals[vals < tol] = tol
    A2 = vecs @ np.diag(vals) @ vecs.T
    D_inv = np.diag(1 / np.sqrt(np.diag(A2)))
    return D_inv @ A2 @ D_inv

# --------------------------------------------------
# General Monte‑Carlo pricer (basket / best / worst)
# --------------------------------------------------

def exotic_mc(
    S0: np.ndarray,
    K: float,
    T: float,
    r: float,
    q: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    payoff_flavour: str = "basket",  # "basket" | "best" | "worst"
    call_put: str = "call",          # "call" | "put"
    weights: Optional[np.ndarray] = None,
    n_sim: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> float:
    """Vectorised MC pricer for basket / best‑of / worst‑of European options."""
    if payoff_flavour == "basket" and weights is None:
        raise ValueError("Weights must be supplied for basket option")

    rng = np.random.default_rng(seed)

    # ensure PD correlation
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr = nearest_pd(corr)
        L = np.linalg.cholesky(corr)

    m = n_sim // 2 if antithetic else n_sim
    Z = rng.standard_normal((m, len(S0)))
    if antithetic:
        Z = np.vstack([Z, -Z])

    X = Z @ L.T  # correlate
    drift = (r - q - 0.5 * sigma ** 2) * T
    vol_t = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + vol_t * X)  # shape (n_sim, n_assets)

    if payoff_flavour == "basket":
        value = ST @ weights
    elif payoff_flavour == "best":
        value = ST.max(axis=1)
    else:  # worst
        value = ST.min(axis=1)

    if call_put == "call":
        payoff = np.maximum(value - K, 0.0)
    else:
        payoff = np.maximum(K - value, 0.0)

    return math.exp(-r * T) * payoff.mean()

# --------------------------------------------------
# Sensitivity cheat‑sheets (directional arrows)
# --------------------------------------------------
ARROW = {+1: "↑", -1: "↓", 0: "-", None: "?"}

SENS_MAP = {
    # (flavour, call_put): (dividend, vol, correlation)
    ("basket", "call"):   (-1, +1, +1),
    ("basket", "put"):    (+1, +1, +1),
    ("worst", "call"):    (-1, None, +1),
    ("worst", "put"):     (+1, +1, -1),
    ("best", "call"):     (-1, +1, -1),
    ("best", "put"):      (+1, None, +1),
}

# ---------------- Forward-impact rule-set -----------------
FWD_MAP = {
    "worst":  {"div": -1, "vol": -1, "corr": +1},
    "best":   {"div": -1, "vol": +1, "corr": -1},
    "basket": {"div": -1, "vol":  0, "corr":  0},
}

# correlation raises basket variance ⇒ +1 for basket, 0 otherwise
CORR_VOL_MAP = {"basket": +1, "worst": 0, "best": 0}

def fwd_arrow(flavour: str, driver: str) -> str:
    """Return the unicode arrow for forward-impact row."""
    return ARROW[FWD_MAP[flavour][driver]]

def corr_vol_arrow(flavour: str) -> str:
    return ARROW[CORR_VOL_MAP[flavour]]

# --------------------------------------------------
# Streamlit user interface
# --------------------------------------------------

def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Multi‑Asset Exotic Option Pricer (Basket / Best / Worst)")

    # ── Market parameters ─────────────────────────────────────────────
    st.sidebar.header("Market Parameters")
    S_spot = st.sidebar.number_input("Spot Price (all assets)", value=100.0, format="%.4f")
    r = st.sidebar.number_input("Risk‑free rate r", value=0.02, format="%.4f")
    K = st.sidebar.number_input("Strike K", value=100.0, format="%.4f")
    T = st.sidebar.number_input("Maturity T (yrs)", value=1.0, format="%.4f")
    n_sim = st.sidebar.number_input("MC paths", min_value=10_000, step=10_000, value=100_000)
    eps = st.sidebar.number_input("Cega bump Δρ", min_value=1e-4, value=0.01, format="%.4f")

    payoff_flavour = st.sidebar.selectbox("Payoff type", ["basket", "best", "worst"], index=0)
    call_put = st.sidebar.selectbox("Option side", ["call", "put"], index=0)

    # ── Asset parameters ──────────────────────────────────────────────
    st.sidebar.header("Asset Parameters (3 assets)")
    spots, vols, divs, wts = [], [], [], []
    for i in range(3):
        spots.append(S_spot)  # same spot for simplicity; could be separate inputs
        vols.append(st.sidebar.number_input(f"Vol σ{i+1}", value=0.20, step=0.01, format="%.4f"))
        divs.append(st.sidebar.number_input(f"Dividend q{i+1}", value=0.03, step=0.005, format="%.4f"))
        if payoff_flavour == "basket":
            wts.append(st.sidebar.number_input(f"Weight w{i+1}", value=1/3, min_value=0.0, max_value=1.0, step=0.01, format="%.4f"))

    if payoff_flavour != "basket":
        wts = [1/3, 1/3, 1/3]  # placeholder; ignored in pricing

    # ── Correlation matrix input ──────────────────────────────────────
    st.sidebar.header("Correlation Matrix (3×3)")
    corr = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            corr_val = st.sidebar.number_input(f"ρ{i+1}{j+1}", min_value=-0.99, max_value=0.99, value=0.0, step=0.01, format="%.2f")
            corr[i, j] = corr[j, i] = corr_val

    # ── Display inputs (main panel) ───────────────────────────────────
    df_assets = pd.DataFrame({"Spot": spots, "Volatility": vols, "Dividend": divs, "Weight": wts}, index=[f"Asset {i+1}" for i in range(3)])
    st.subheader("📊 Asset Parameters")
    st.table(df_assets)

    st.subheader("📈 Correlation Matrix")
    st.table(pd.DataFrame(corr, index=df_assets.index, columns=df_assets.index))

    # ── Always‑on sensitivities ───────────────────────────────────────
    # --------- pick the column title dynamically ------------
    fwd_label = f"Forward of {payoff_flavour}"   # basket / worst / best
    vol_label = f"Vol of {payoff_flavour}" 

    div_dir, vol_dir, cor_dir = SENS_MAP[(payoff_flavour, call_put)]

    table_md = f"""
      <table style='border:1px solid #ccc;border-collapse:collapse;width:60%;margin-top:1rem;'>
        <tr><th colspan='4' style='border:1px solid #ccc;padding:8px;text-align:center;'>
            {call_put.capitalize()} Price Sensitivity — {payoff_flavour}</th></tr>

        <tr><th style='border:1px solid #ccc;padding:6px;'></th>
            <th style='border:1px solid #ccc;padding:6px;text-align:center;'>{fwd_label}</th>
            <th style='border:1px solid #ccc;padding:6px;text-align:center;'>{vol_label}</th>
            <th style='border:1px solid #ccc;padding:6px;text-align:center;'>Total impact</th></tr>

        <!-- Dividend row -->
        <tr><td style='border:1px solid #ccc;padding:6px;'>Dividend ↑</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{ARROW[-1]}</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>-</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{ARROW[div_dir]}</td></tr>

        <!-- Volatility row -->
        <tr><td style='border:1px solid #ccc;padding:6px;'>Volatility ↑</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{fwd_arrow(payoff_flavour,'vol')}</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{ARROW[+1]}</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{ARROW[vol_dir]}</td></tr>

        <!-- Correlation row -->
        <tr><td style='border:1px solid #ccc;padding:6px;'>Correlation ↑</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{fwd_arrow(payoff_flavour,'corr')}</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{corr_vol_arrow(payoff_flavour)}</td>
            <td style='border:1px solid #ccc;padding:6px;text-align:center;'>{ARROW[cor_dir]}</td></tr>
      </table>"""


    st.markdown(table_md, unsafe_allow_html=True)

    # ── Compute button ────────────────────────────────────────────────
    if st.sidebar.button("Compute Price & Cega"):
        S0_arr = np.array(spots)
        sigma = np.array(vols)
        q_arr = np.array(divs)
        wts_arr = np.array(wts)

        # Pricing
        price_base = exotic_mc(S0_arr, K, T, r, q_arr, sigma, corr, payoff_flavour, call_put, wts_arr, int(n_sim), seed=42)

        # Cega (average over pairwise bumps)
        derivs = []
        for i in range(3):
            for j in range(i + 1, 3):
                corr_up = corr.copy()
                corr_up[i, j] = corr_up[j, i] = min(corr[i, j] + eps, 0.999)
                p_up = exotic_mc(S0_arr, K, T, r, q_arr, sigma, corr_up, payoff_flavour, call_put, wts_arr, int(n_sim), seed=42)
                derivs.append((p_up - price_base) / eps)
        cega_val = float(np.mean(derivs))

        # Results
        st.subheader("🍀 Results")
        st.write(f"**{call_put.capitalize()} price ({payoff_flavour})** : {price_base:.4f}")
        st.write(f"**Cega**                         : {cega_val:.6f}")

    


if __name__ == "__main__":
    main()
