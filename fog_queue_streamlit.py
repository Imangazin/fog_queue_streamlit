# Compatible Streamlit app for Python 3.6 / Streamlit 1.1
# Run: streamlit run fog_queue_streamlit_legacy.py --server.port 8501 --server.address 0.0.0.0

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Fog Computing Queueing Model", layout="wide")
st.title("Queueing Theory Model for Fog Computing")

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Input Parameters")

J = st.sidebar.slider("Number of Jobs (J)", 10, 300, 100, 10)
mu_E = st.sidebar.number_input("μE (Entry Server)", 0.05, 5.0, 0.9, 0.05)
mu_P = st.sidebar.number_input("μP (Processing Server)", 0.05, 5.0, 0.4, 0.05)
mu_D = st.sidebar.number_input("μD (Database Server)", 0.05, 5.0, 0.4, 0.05)
mu_O = st.sidebar.number_input("μO (Output Server)", 0.05, 5.0, 0.4, 0.05)
mu_F = st.sidebar.number_input("μF (Fog Server)", 0.05, 5.0, 0.4, 0.05)
mu_C = st.sidebar.number_input("μC (Client Server)", 0.05, 5.0, 0.4, 0.05)

R = st.sidebar.number_input("Processing Servers R", 1, 200, 10)
N = st.sidebar.number_input("Fog Servers N", 1, 200, 10)
M = st.sidebar.number_input("Client Servers M", 1, 200, 10)

delta = st.sidebar.slider("δ (DB access probability)", 0.0, 1.0, 0.5, 0.05)
tau = st.sidebar.slider("τ (Output server probability)", 0.0, 1.0, 0.5, 0.05)
kappa = st.sidebar.slider("κ (Fog exit probability)", 0.0, 1.0, 0.5, 0.05)
SLA = st.sidebar.number_input("SLA (seconds)", 0.1, 30.0, 4.0, 0.1)

run_sim = st.sidebar.button("Calculate")

nodes = ["ES", "PS", "DS", "OS", "CS", "FS"]

# --------------------------
# Helper functions
# --------------------------
def visit_ratios_from_T(T):
    w, vecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(w - 1))
    v = np.real(vecs[:, idx])
    v = np.abs(v)
    if v[0] == 0: v[0] = 1
    return v / v[0]

def mva_schweitzer(J, V, D, m):
    K = len(D)
    Q_prev = np.zeros(K)
    X_hist, T_hist = [], []
    for n in range(1, J + 1):
        Rn = np.zeros(K)
        for k in range(K):
            if m[k] <= 1:
                Rn[k] = D[k] * (1 + Q_prev[k])
            else:
                Rn[k] = D[k] * (1 + Q_prev[k] / m[k])
        Tn = np.sum(V * Rn)
        Xn = n / Tn
        Qn = Xn * V * Rn
        Q_prev = Qn
        X_hist.append(Xn)
        T_hist.append(Tn)
    return np.array(X_hist), np.array(T_hist)

# --------------------------
# Section selection (instead of tabs)
# --------------------------
section = st.radio("Select section:", ["Diagram", "Performance", "Summary"])

# --------------------------
# DIAGRAM
# --------------------------
if section == "Diagram":
    st.subheader("System Diagram")
    if os.path.exists("diagram.png"):
        try:
            st.image("diagram.png", use_column_width=True,
                     caption="Fog computing queueing network", clamp=True)
        except Exception as e:
            st.warning("⚠️ Could not load image: %s" % e)
    else:
        st.info("📷 Place a file named **diagram.png** next to this script to display it here.")

# --------------------------
# PERFORMANCE
# --------------------------
elif section == "Performance":
    if not run_sim:
        st.info("Adjust parameters in the sidebar and click **Calculate** to generate results.")
    else:
        # Build transition matrix (same as paper)
        T = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, (1 - delta)*(1 - tau), delta, (1 - delta)*tau, 0, 0],
            [0, (1 - tau), 0, tau, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [kappa, 0, 0, 0, (1 - kappa), 0],
        ])

        V = visit_ratios_from_T(T)
        m = np.array([1, R, 1, 1, M, N], dtype=float)
        mu = np.array([mu_E, mu_P, mu_D, mu_O, mu_C, mu_F], dtype=float)
        D = 1.0 / mu

        X_hist, T_hist = mva_schweitzer(J, V, D, m)
        jobs = np.arange(1, J + 1)
        util = np.clip((X_hist[-1] * V * D) / m, 0, 1)
        J_sla = int(np.max(jobs[T_hist <= SLA])) if np.any(T_hist <= SLA) else 0

        st.subheader("Throughput vs Jobs")
        fig1, ax1 = plt.subplots()
        ax1.plot(jobs, X_hist, lw=2)
        ax1.set_xlabel("Number of jobs (J)")
        ax1.set_ylabel("Throughput (jobs/s)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig1)

        st.subheader("Response Time vs Jobs")
        fig2, ax2 = plt.subplots()
        ax2.plot(jobs, T_hist, lw=2, label="Response time")
        ax2.axhline(SLA, color="red", linestyle="--", label="SLA = %.1fs" % SLA)
        if J_sla > 0:
            ax2.axvline(J_sla, linestyle=":", label="Max J ≤ SLA ≈ %d" % J_sla)
        ax2.legend()
        ax2.set_xlabel("Number of jobs (J)")
        ax2.set_ylabel("Response time (s)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)

        st.subheader("Node Utilization (ρ)")
        fig3, ax3 = plt.subplots()
        ax3.bar(nodes, util)
        ax3.set_ylim(0, 1.05)
        for i, u in enumerate(util):
            ax3.text(i, min(1.02, u + 0.02), "%.2f" % u, ha="center", va="bottom")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig3)

# --------------------------
# SUMMARY
# --------------------------
else:
    if not run_sim:
        st.info("Run the simulation first to view summary results.")
    else:
        T = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, (1 - delta)*(1 - tau), delta, (1 - delta)*tau, 0, 0],
            [0, (1 - tau), 0, tau, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [kappa, 0, 0, 0, (1 - kappa), 0],
        ])
        V = visit_ratios_from_T(T)
        m = np.array([1, R, 1, 1, M, N], dtype=float)
        mu = np.array([mu_E, mu_P, mu_D, mu_O, mu_C, mu_F], dtype=float)
        D = 1.0 / mu

        X_hist, T_hist = mva_schweitzer(J, V, D, m)
        util = np.clip((X_hist[-1] * V * D) / m, 0, 1)

        df = pd.DataFrame({
            "Node": nodes,
            "Servers": m.astype(int),
            "Service rate μ": mu,
            "Visit ratio V": np.round(V, 4),
            "Service demand D (s)": np.round(D, 4),
            "Utilization ρ": np.round(util, 4)
        })
        st.subheader("Summary Table")
        st.write(df)

        # Save summary locally
        df.to_csv("fog_model_summary.csv", index=False)
        st.success("Summary saved to fog_model_summary.csv")
