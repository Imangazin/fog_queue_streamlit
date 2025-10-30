#!/usr/bin/env python3
"""
Fog Computing Queueing Theory Model (Streamlit)
Ported from R Shiny version using Schweitzer MVA approximation.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Fog Computing Queueing Model", layout="wide")
st.title("Queueing Theory Model for Fog Computing")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def visit_ratios_from_T(T: np.ndarray) -> np.ndarray:
    """Compute normalized visit ratios V from routing matrix T."""
    w, vecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(w - 1.0))
    v = np.real(vecs[:, idx])
    if np.allclose(v, 0):
        v = np.ones(T.shape[0])
    v = np.abs(v)
    return v / (v[0] if v[0] != 0 else 1.0)

def mva_schweitzer(J: int, V: np.ndarray, D: np.ndarray, m: np.ndarray):
    """Approximate Schweitzer MVA for multi-server closed queueing networks."""
    K = len(D)
    Q_prev = np.zeros(K)
    X_hist, T_hist, R_hist, Q_hist = [], [], [], []

    for n in range(1, J + 1):
        Rn = np.zeros(K)
        for k in range(K):
            if m[k] <= 1:
                Rn[k] = D[k] * (1.0 + Q_prev[k])
            else:
                Rn[k] = D[k] * (1.0 + Q_prev[k] / m[k])
        Tn = float(np.sum(V * Rn))
        Xn = n / Tn
        Qn = Xn * V * Rn
        X_hist.append(Xn)
        T_hist.append(Tn)
        R_hist.append(Rn)
        Q_hist.append(Qn)
        Q_prev = Qn
    return np.array(X_hist), np.array(T_hist), np.array(R_hist), np.array(Q_hist)

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    J = st.slider("Number of Jobs (J)", 10, 300, 100, step=10)
    colA, colB = st.columns(2)
    with colA:
        mu_E = st.number_input("μE (ES)", 0.05, 5.0, 0.9, 0.05)
        mu_P = st.number_input("μP (PS)", 0.05, 5.0, 0.4, 0.05)
        mu_D = st.number_input("μD (DS)", 0.05, 5.0, 0.4, 0.05)
    with colB:
        mu_O = st.number_input("μO (OS)", 0.05, 5.0, 0.4, 0.05)
        mu_F = st.number_input("μF (FS)", 0.05, 5.0, 0.4, 0.05)
        mu_C = st.number_input("μC (CS)", 0.05, 5.0, 0.4, 0.05)

    R = st.number_input("Processing servers R (PS)", 1, 200, 10)
    N = st.number_input("Fog servers N (FS)", 1, 200, 10)
    M = st.number_input("Client servers M (CS)", 1, 200, 10)

    delta = st.slider("δ — DB access probability", 0.0, 1.0, 0.5, 0.05)
    tau   = st.slider("τ — Output server probability", 0.0, 1.0, 0.5, 0.05)
    kappa = st.slider("κ — Fog exit probability", 0.0, 1.0, 0.5, 0.05)
    SLA = st.number_input("SLA (seconds)", 0.1, 30.0, 4.0, 0.1)

    run_sim = st.button("Run Simulation", use_container_width=True)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Diagram", "Performance Plots", "Summary"])

with tab1:
    st.subheader("System Diagram")
    st.caption("Place a file named **diagram.png** next to this script to display it below.")
    st.image("diagram.png", use_column_width=True, caption="Fog computing queueing network", clamp=True)

# ------------------------------------------------------------
# Simulation logic
# ------------------------------------------------------------
if not run_sim:
    with tab2:
        st.info("Adjust inputs in the sidebar and click **Run Simulation** to generate results.")
else:
    # Routing matrix
    T = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, (1 - delta)*(1 - tau), delta, (1 - delta)*tau, 0, 0],
        [0, (1 - tau), 0, tau, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [kappa, 0, 0, 0, (1 - kappa), 0]
    ], dtype=float)

    # Compute visit ratios
    V = visit_ratios_from_T(T)
    m = np.array([1, R, 1, 1, M, N], dtype=float)

    # Match R scaling for fairness
    mu_E *= 2.0
    mu_D *= 3.5
    mu_O *= 3.5
    mu_values = np.array([mu_E, mu_P, mu_D, mu_O, mu_C, mu_F], dtype=float)
    D = 1.0 / mu_values

    # Run Schweitzer MVA
    X_hist, T_hist, R_hist, Q_hist = mva_schweitzer(J, V, D, m)
    jobs = np.arange(1, J + 1)
    util = np.clip((X_hist[-1] * V * D) / m, 0.0, 1.0)
    Lk = Q_hist[-1]
    J_sla = int(np.max(jobs[T_hist <= SLA])) if np.any(T_hist <= SLA) else 0

    # ------------------------------------------------------------
    # Performance Plots
    # ------------------------------------------------------------
    with tab2:
        col1, col2 = st.columns(2)

        # Throughput plot
        with col1:
            st.subheader("Throughput evolution")
            fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
            ax1.plot(jobs, X_hist, 'b-', linewidth=2, label="Throughput")
            ax1.scatter(jobs, X_hist, color='dodgerblue', s=15)
            ax1.set_xlabel("# Jobs")
            ax1.set_ylabel("Throughput (jobs/time unit)")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.35)
            st.pyplot(fig1, clear_figure=True)

        # Mean time (Response time) plot
        with col2:
            st.subheader("Mean time evolution")
            fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
            ax2.plot(jobs, T_hist, 'b-', linewidth=2, label="Mean time")
            ax2.scatter(jobs, T_hist, color='dodgerblue', s=15)
            ax2.axhline(SLA, linestyle="--", color="red", linewidth=2, label=f"SLA = {SLA:.1f}s")
            ax2.text(2, SLA + 0.1, "SLA Limit", color="red", fontsize=10)
            ax2.set_xlabel("# Jobs")
            ax2.set_ylabel("Mean time (s)")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.35)
            st.pyplot(fig2, clear_figure=True)

        # Node utilization
        st.subheader("Node Usage (ρ)")
        fig3, ax3 = plt.subplots(figsize=(13, 4.2))
        nodes = ["Nd1 (ES)", "Nd2 (PS)", "Nd3 (DS)", "Nd4 (OS)", "Nd5 (CS)", "Nd6 (FS)"]
        ax3.bar(nodes, util, color="skyblue", edgecolor="gray")
        for i, u in enumerate(util):
            ax3.text(i, min(1.02, u + 0.02), f"{u:.2f}", ha="center", va="bottom")
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("Utilization ρ")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.25)
        st.pyplot(fig3, clear_figure=True)

    # ------------------------------------------------------------
    # Summary (tab3)
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Summary (Comparable to R QueueingModel)")
        df = pd.DataFrame({
            "Node": nodes,
            "Servers (m_k)": m.astype(int),
            "Service rate μ": np.round(mu_values, 3),
            "Visit ratio V": np.round(V, 3),
            "Service demand D (s)": np.round(D, 4),
            "Utilization ρ": np.round(util, 4),
            "Mean customers Lk": np.round(Lk, 4)
        })
        st.dataframe(df, use_container_width=True)
