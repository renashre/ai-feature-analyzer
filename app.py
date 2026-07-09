import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from experiment import simulate_experiment, FEATURES
from analysis import (
    calculate_engagement,
    calculate_retention,
    calculate_user_quality,
    calculate_statistical_significance,
    calculate_novelty_effect
)
from verdict import generate_verdict

st.set_page_config(
    page_title="AI Feature Impact Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔬 AI Feature Impact Analyzer")
st.markdown("*Evaluating whether AI features genuinely improve products — or just create short-term engagement spikes.*")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment Settings")
    selected_feature = st.selectbox("Select AI Feature", FEATURES)
    launch_date = st.date_input("Launch Date", value=pd.Timestamp("2017-01-01"))
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**What this tool evaluates:**")
    st.markdown("- Engagement lift")
    st.markdown("- Retention curves")
    st.markdown("- User quality")
    st.markdown("- Statistical significance")
    st.markdown("- Novelty effect detection")

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# ── Run analysis ─────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Running experiment simulation..."):
        df_exp = simulate_experiment(df, feature_name=selected_feature, launch_date=str(launch_date))
        engagement   = calculate_engagement(df_exp)
        retention    = calculate_retention(df_exp)
        user_quality = calculate_user_quality(df_exp)
        stats        = calculate_statistical_significance(df_exp)
        novelty      = calculate_novelty_effect(df_exp)
        verdict      = generate_verdict(
            feature_name=selected_feature,
            engagement=engagement,
            retention=retention,
            user_quality=user_quality,
            stats=stats,
            novelty=novelty
        )

    # ── Verdict card ─────────────────────────────────────────────────────────
    st.subheader("📋 Verdict")
    st.code(verdict, language=None)
    st.divider()

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Avg Session Duration",
            value=f"{stats['treatment_mean']:.1f}s",
            delta=f"{stats['treatment_mean'] - stats['control_mean']:.2f}s vs control"
        )
    with col2:
        treatment_conv = engagement[engagement["group"] == "treatment"]["conversion_rate"].values[0]
        control_conv   = engagement[engagement["group"] == "control"]["conversion_rate"].values[0]
        st.metric(
            label="Conversion Rate",
            value=f"{treatment_conv:.2%}",
            delta=f"{treatment_conv - control_conv:.2%} vs control"
        )
    with col3:
        treatment_quality = user_quality[user_quality["group"] == "treatment"]["avg_user_quality_score"].values[0]
        control_quality   = user_quality[user_quality["group"] == "control"]["avg_user_quality_score"].values[0]
        st.metric(
            label="User Quality Score",
            value=f"{treatment_quality:.0f}",
            delta=f"{treatment_quality - control_quality:.0f} vs control"
        )
    with col4:
        st.metric(
            label="P-Value",
            value=f"{stats['p_value']:.4f}",
            delta="significant" if stats["p_value"] < 0.05 else "not significant",
            delta_color="normal" if stats["p_value"] < 0.05 else "inverse"
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Retention curves
    with col_left:
        st.subheader("📈 Retention Curves")
        fig, ax = plt.subplots(figsize=(6, 4))
        for group, color in [("control", "#888780"), ("treatment", "#185FA5")]:
            group_data = retention[retention["group"] == group]
            ax.plot(
                group_data["weeks_since_launch"],
                group_data["retention_rate"],
                label=group.capitalize(),
                color=color,
                linewidth=2
            )
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Weeks Since Launch")
        ax.set_ylabel("Retention Rate")
        ax.set_title("Retention: Control vs Treatment")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # Novelty effect
    with col_right:
        st.subheader("🔍 Novelty Effect")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            novelty["weeks_since_launch"],
            novelty["treatment_lift"],
            color="#185FA5",
            linewidth=2,
            label="Treatment lift"
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.fill_between(
            novelty["weeks_since_launch"],
            novelty["treatment_lift"],
            0,
            where=novelty["treatment_lift"] > 0,
            alpha=0.1,
            color="#185FA5"
        )
        ax.fill_between(
            novelty["weeks_since_launch"],
            novelty["treatment_lift"],
            0,
            where=novelty["treatment_lift"] < 0,
            alpha=0.1,
            color="#E24B4A"
        )
        ax.set_xlabel("Weeks Since Launch")
        ax.set_ylabel("Session Duration Lift (seconds)")
        ax.set_title("Treatment Lift Over Time")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Engagement bar chart
    st.subheader("⚡ Engagement Breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        groups = engagement["group"].str.capitalize()
        values = engagement["avg_session_duration"]
        bars = ax.bar(groups, values, color=["#888780", "#185FA5"], width=0.4)
        ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=10)
        ax.set_ylabel("Avg Session Duration (s)")
        ax.set_title("Session Duration by Group")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3))
        values = engagement["avg_pages_visited"]
        bars = ax.bar(groups, values, color=["#888780", "#185FA5"], width=0.4)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
        ax.set_ylabel("Avg Pages Visited")
        ax.set_title("Pages Visited by Group")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

else:
    st.info("👈 Select an AI feature in the sidebar and click **Run Analysis** to get started.")