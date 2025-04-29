import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# ═══════════════════════════════════════════════════════
#  Turnbull‑Wakeman (geometric‑avg, discrete fixings)
# ═══════════════════════════════════════════════════════

def tw_geo_asian_call(S0, K, r, q, sigma, T, n):
    a1 = (n + 1) / (2 * n)
    a2 = (n + 1) * (2 * n + 1) / (6 * n ** 2)
    mu    = (r - q - 0.5 * sigma ** 2)
    mu_g  = (mu + 0.5 * sigma ** 2) * a1 * T
    var_g = sigma ** 2 * a2 * T
    sigma_g = math.sqrt(var_g)
    d1 = (math.log(S0 / K) + mu_g + 0.5 * var_g) / sigma_g
    d2 = d1 - sigma_g
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    disc = math.exp(-r * T)
    Fg   = S0 * math.exp(mu_g)
    return disc * (Fg * N(d1) - K * N(d2))

# ═══════════════════════════════════════════════════════
#  MC arithmetic‑avg Asian call + Greeks (Δ Γ Vega Θ ρ)
# ═══════════════════════════════════════════════════════

def mc_arith_call(S0, K, r, q, sigma, T, n, sims, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol   = sigma * math.sqrt(dt)
    Z = rng.standard_normal((sims, n))
    paths = S0 * np.exp(np.cumsum(drift + vol * Z, axis=1))
    avg   = paths.mean(axis=1)
    payoff = np.maximum(avg - K, 0.0)
    return math.exp(-r * T) * payoff.mean()


def greeks_bump(S0, K, r, q, sigma, T, n, sims,
                dS=1.0, dSigma=1e-4, dT=1/365, dR=1e-4, seed=0):
    price = lambda **kw: mc_arith_call(seed=seed, **kw)
    base = price(S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, n=n, sims=sims)
    pu   = price(S0=S0+dS, K=K, r=r, q=q, sigma=sigma, T=T, n=n, sims=sims)
    pd   = price(S0=S0-dS, K=K, r=r, q=q, sigma=sigma, T=T, n=n, sims=sims)
    du   = price(S0=S0, K=K, r=r, q=q, sigma=sigma+dSigma, T=T, n=n, sims=sims)
    dd   = price(S0=S0, K=K, r=r, q=q, sigma=sigma-dSigma, T=T, n=n, sims=sims)
    tm   = price(S0=S0, K=K, r=r, q=q, sigma=sigma, T=max(T-dT,1e-6), n=n, sims=sims)
    ru   = price(S0=S0, K=K, r=r+dR, q=q, sigma=sigma, T=T, n=n, sims=sims)
    rd   = price(S0=S0, K=K, r=r-dR, q=q, sigma=sigma, T=T, n=n, sims=sims)
    delta = (pu - pd) / (2 * dS)*100
    gamma = (pu - 2*base + pd) / dS**2*10000
    vega  = (du - dd) / (2 * dSigma)/100
    theta = (tm - base) / (dT)/100
    rho   = (ru - rd) / (2 * dR)/10000
    return base, delta, gamma, vega, theta, rho

# ═══════════════════════════════════════════════════════
#  Streamlit App
# ═══════════════════════════════════════════════════════

def main():
    st.set_page_config(layout="wide")
    st.title("Asian Call – Arithmetic vs Geometric (Greeks & Plots)")

    sb = st.sidebar
    sb.header("Market inputs")
    S0 = sb.number_input("Spot S₀", value=6056.78)
    K  = sb.number_input("Strike K", value=6056.78)
    r  = sb.number_input("r (cont)", value=0.04348, format="%.5f")
    q  = sb.number_input("q (cont)", value=0.01320, format="%.5f")
    sigma = sb.number_input("Vol σ", value=0.13835, format="%.5f")
    Tdays = sb.number_input("T (days)", value=90)
    T = Tdays / 365.0

    sb.header("Fixing frequency")
    freq = sb.selectbox("Choose averaging frequency", ("Weekly","Monthly","Semi‑Annual","Annual","Custom"))
    freq_map = {"Weekly":52,"Monthly":12,"Semi‑Annual":2,"Annual":1}
    if freq == "Custom":
        n_fix = sb.number_input("Custom fixings", value=12, min_value=1, step=1)
    else:
   
        n_fix = max(1, round(freq_map[freq] * T))      
        sb.write(f"→ Using **{n_fix}** fixings over {Tdays:.0f} days")

    sb.header("MC settings")
    sims = sb.number_input("MC paths", value=100_000, step=20000)
    seed = sb.number_input("Seed", value=12345)

    if sb.button("Compute & plot"):
        geo_price = tw_geo_asian_call(S0, K, r, q, sigma, T, n_fix)
        ar_price, d, g, v, th, rh = greeks_bump(S0, K, r, q, sigma, T, n_fix, int(sims), seed=seed)

        c1, c2 = st.columns(2)
        c1.metric("Geometric price", f"{geo_price:.4f}")
        c2.metric("Arithmetic price", f"{ar_price:.4f}")

        st.table(pd.DataFrame({"Greek":["Delta","Gamma","Vega","Theta","Rho"],
                               "Value":[d,g,v,th,rh]}))

        # sweep plots
        S_grid = np.linspace(0.7*S0, 1.3*S0, 31)
        series = {"Price":[],"Delta":[],"Gamma":[],"Vega":[],"Theta":[],"Rho":[]}
        for s in S_grid:
            p_,d_,g_,v_,th_,rh_ = greeks_bump(s, K, r, q, sigma, T, n_fix, int(sims//4), seed=seed)
            for arr,val in zip(series.values(),[p_,d_,g_,v_,th_,rh_]):
                arr.append(val)

        fig, axes = plt.subplots(3,2, figsize=(10,9))
        for ax,(name,vals) in zip(axes.flatten(), series.items()):
            ax.plot(S_grid, vals)
            ax.set_title(name)
            ax.set_xlabel("Spot")
        fig.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
