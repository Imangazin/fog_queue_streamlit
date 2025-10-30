# Streamlit app: Queueing Theory Model for Fog Computing (R/Shiny parity)
# Run: streamlit run fog_queue_streamlit.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Page + layout
# ---------------------------
st.set_page_config(page_title="Fog Computing Queueing Model", layout="wide")
st.title("Queueing Theory Model for Fog Computing")

# ---------------------------
# Helpers
# ---------------------------
def visit_ratios_from_T(T: np.ndarray) -> np.ndarray:
    """Return visit ratios V (normalize so ES=1) for left stationary vector v: v = vT."""
    w, vecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(w - 1.0))
    v = np.real(vecs[:, idx])
    if np.allclose(v, 0):
        v = np.ones(T.shape[0])
    v = np.abs(v)
    V = v / (v[0] if v[0] != 0 else 1.0)
    return V

def mva_schweitzer(J: int, V: np.ndarray, D: np.ndarray, m: np.ndarray):
    """Single-class Schweitzer approximate MVA for multi-server centers."""
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
    return (np.array(X_hist), np.array(T_hist),
            np.array(R_hist), np.array(Q_hist))

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Inputs")
    J = st.slider("Number of Jobs (J)", min_value=10, max_value=300, value=100, step=10)
    colA, colB = st.columns(2)
    with colA:
        mu_E = st.number_input("μE (ES) Service rate of the Entry Server", min_value=0.05, max_value=5.0, value=0.9, step=0.05)
        mu_P = st.number_input("μP (PS) Service rate of every Processing Server", min_value=0.05, max_value=5.0, value=0.4, step=0.05)
        mu_D = st.number_input("μD (DS) Service rate of the Database Server", min_value=0.05, max_value=5.0, value=0.4, step=0.05)
    with colB:
        mu_O = st.number_input("μO (OS) Service rate of the Output Server", min_value=0.05, max_value=5.0, value=0.4, step=0.05)
        mu_F = st.number_input("μF (FS) Service rate of every fog Server", min_value=0.05, max_value=5.0, value=0.4, step=0.05)
        mu_C = st.number_input("μC (CS) Service rate of every Client Server", min_value=0.05, max_value=5.0, value=0.4, step=0.05)

    R = st.number_input("Processing servers R (PS)", min_value=1, max_value=200, value=10)
    N = st.number_input("Fog servers N (FS)", min_value=1, max_value=200, value=10)
    M = st.number_input("Client servers M (CS)", min_value=1, max_value=200, value=10)

    delta = st.slider("δ — DB access probability", 0.0, 1.0, 0.5, 0.05)
    tau   = st.slider("τ — Output server probability", 0.0, 1.0, 0.5, 0.05)
    kappa = st.slider("κ — Fog exit probability",    0.0, 1.0, 0.5, 0.05)

    SLA = st.number_input("SLA (service-level agreement) (seconds)", min_value=0.1, max_value=30.0, value=4.0, step=0.1)
    run_sim = st.button("Calculate", use_container_width=True)

nodes = ["ES","PS","DS","OS","CS","FS"]

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Diagram", "Performance", "Summary"])

with tab1:
    st.subheader("System Diagram")
    st.caption("Place a file named **diagram.png** next to this script to display it below.")
    st.image("diagram.png", use_column_width=True, caption="Fog computing queueing network", clamp=True)

with tab2:
    if not run_sim:
        st.info("Adjust inputs in the sidebar and click **Calculate** to generate performance plots.")
    else:
        # Build transition matrix (same topology/order as R/Shiny)
        # Order: [ES, PS, DS, OS, CS, FS]
        T = np.array([
            [0, 1, 0, 0, 0, 0],                                       # ES -> PS
            [0, (1 - delta)*(1 - tau), delta, (1 - delta)*tau, 0, 0], # PS
            [0, (1 - tau), 0, tau, 0, 0],                             # DS
            [0, 0, 0, 0, 1, 0],                                       # OS -> CS
            [0, 0, 0, 0, 0, 1],                                       # CS -> FS
            [kappa, 0, 0, 0, (1 - kappa), 0],                         # FS -> ES or CS
        ], dtype=float)

        V = visit_ratios_from_T(T)
        m = np.array([1, R, 1, 1, M, N], dtype=float)

        # --- R/Shiny parity: service-rate scaling ---
        # R code:
        # es_service_rate <- input$mu_es * 2
        # ds_service_rate <- input$mu_ds * 3.5
        # os_service_rate <- input$mu_os * 3.5
        mu = np.array([
            mu_E * 2.0,   # ES
            mu_P,         # PS
            mu_D * 3.5,   # DS
            mu_O * 3.5,   # OS
            mu_C,         # CS
            mu_F          # FS
        ], dtype=float)
        D = 1.0 / mu

        X_hist, T_hist, R_hist, Q_hist = mva_schweitzer(J, V, D, m)
        jobs = np.arange(1, J + 1)

        # Utilization ρ_k at final J; ROk analogue in R
        util = (X_hist[-1] * V * D) / m
        util = np.clip(util, 0.0, 1.0)

        # Mean customers per node (Lk) at final J; matches R m_cjn1$Lk
        Lk_final = Q_hist[-1]  # Qn is mean number at each center for that population

        # J within SLA (use last J satisfying T_hist <= SLA)
        J_sla = int(np.max(jobs[T_hist <= SLA])) if np.any(T_hist <= SLA) else 0

        # --- Throughput (R: points + smooth); no capacity/asymptote line ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Throughput evolution")
            fig1, ax1 = plt.subplots(figsize=(6.5,4.2))
            ax1.plot(jobs, X_hist, marker='o', markersize=3, linewidth=1.5, label="Throughput")
            ax1.set_xlabel("# Jobs")
            ax1.set_ylabel("Throughput (jobs/time unit)")
            ax1.grid(True, linestyle="--", alpha=0.35)
            ax1.legend()
            st.pyplot(fig1, clear_figure=True)
            st.download_button("Download throughput plot (PNG)", data=fig_to_bytes(fig1),
                               file_name="throughput_vs_jobs.png", mime="image/png")

        # --- Response time (R: dashed red SLA + label) ---
        with col2:
            st.subheader("Mean time evolution")
            fig2, ax2 = plt.subplots(figsize=(6.5,4.2))
            ax2.plot(jobs, T_hist, marker='o', markersize=3, linewidth=1.5, label="Mean time")
            ax2.axhline(SLA, linestyle="--", color="red", linewidth=2, label=f"SLA = {SLA:.1f}s")

            # Add a small label near SLA line (left side)
            try:
                ax2.text(jobs[0], SLA, "SLA Limit", va="bottom", ha="left")
            except Exception:
                pass

            # Dynamic y-limit and 2-sec ticks (keeps curve visible; matches your desired ticks)
            ymax = float(np.ceil(np.max(T_hist) / 2.0) * 2.0)
            if ymax <= 0:
                ymax = 2.0
            ax2.set_ylim(0, ymax)
            ax2.set_yticks(np.arange(0, ymax + 0.1, 2))

            if J_sla > 0:
                ax2.axvline(J_sla, color="gray", linestyle=":", label=f"Max J within SLA ≈ {J_sla}")

            ax2.set_xlabel("# Jobs")
            ax2.set_ylabel("Mean time (s)")
            ax2.grid(True, linestyle="--", alpha=0.35)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)
            st.download_button("Download response-time plot (PNG)", data=fig_to_bytes(fig2),
                               file_name="response_time_vs_jobs.png", mime="image/png")

        # --- Mean customers per node (R: "Mean customers evolution") ---
        st.subheader("Mean customers evolution")
        fig_mc, ax_mc = plt.subplots(figsize=(13,4.2))
        ax_mc.bar(nodes, Lk_final)
        for i, val in enumerate(Lk_final):
            ax_mc.text(i, val + max(0.02 * np.max(Lk_final), 0.02), f"{val:.2f}", ha="center", va="bottom")
        ax_mc.set_ylabel("Mean jobs")
        ax_mc.grid(True, axis="y", linestyle="--", alpha=0.25)
        st.pyplot(fig_mc, clear_figure=True)
        st.download_button("Download mean-customers plot (PNG)", data=fig_to_bytes(fig_mc),
                           file_name="mean_customers.png", mime="image/png")

        # --- Utilization (ROk in R) ---
        st.subheader("Node Usage")
        fig3, ax3 = plt.subplots(figsize=(13,4.2))
        ax3.bar(nodes, util)
        ax3.set_ylim(0, 1.05)
        for i, u in enumerate(util):
            ax3.text(i, min(1.02, u + 0.02), f"{u:.2f}", ha="center", va="bottom")
        ax3.set_ylabel("Usage (ρ)")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.25)
        st.pyplot(fig3, clear_figure=True)
        st.download_button("Download utilization plot (PNG)", data=fig_to_bytes(fig3),
                           file_name="node_utilization.png", mime="image/png")

with tab3:
    if not run_sim:
        st.info("Run a simulation to see the summary.")
    else:
        st.subheader("Summary (at final population J)")
        # Use displayed (scaled) μ for consistency with R
        mu_used = np.array([mu_E*2.0, mu_P, mu_D*3.5, mu_O*3.5, mu_C, mu_F], dtype=float)
        df = pd.DataFrame({
            "Node": nodes,
            "Servers (m_k)": m.astype(int),
            "Service rate μ": np.round(mu_used, 4),
            "Visit ratio V": np.round(V, 4),
            "Service demand D (s)": np.round(1.0 / mu_used, 4),
            "Utilization ρ": np.round(util, 4),
            "Mean customers Lk": np.round(Lk_final, 4),
        })
        st.dataframe(df, use_container_width=True)
        st.download_button("Download summary CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="fog_model_summary.csv", mime="text/csv")
