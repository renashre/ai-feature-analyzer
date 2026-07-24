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

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 AI Feature Impact Analyzer")
st.markdown("*Evaluating whether AI features genuinely improve products — or just create short-term engagement spikes.*")

# ── FIX 2: How does this work? expander ──────────────────────────────────────
with st.expander("ℹ️ How does this work? (click to expand)"):
    st.markdown("""
    ### The business question
    Every time a company like Meta, Google, or Adobe launches an AI feature, 
    they need to answer one hard question: **did this feature actually make the product better?**
    
    This tool answers that question automatically.
    
    ### How the experiment works
    Rather than showing a new feature to everyone at once, product teams split 
    users into two groups:
    
    | Group | What they see | Purpose |
    |-------|--------------|---------|
    | **Control** | Normal product, no AI feature | The baseline — what would have happened anyway |
    | **Treatment** | Product with the AI feature enabled | The test — did behavior change? |
    
    After the launch, we compare the two groups. If treatment users behave 
    meaningfully better than control users — and the difference is statistically 
    real — the feature is worth shipping to everyone.
    
    ### The five questions this tool answers
    
    | Question | What it means in plain English |
    |----------|-------------------------------|
    | **Engagement lift** | Did treated users spend more time on the product? |
    | **Retention** | Did treated users come back more in the following weeks? |
    | **User quality** | Did the feature attract genuinely interested users, or just empty clicks? |
    | **Statistical significance** | Is the difference real, or could it just be random chance? |
    | **Novelty effect** | Did users engage because the feature is useful, or just because it was new? |
    
    ### How to read the verdict
    - ✅ **Ship** — strong evidence the feature is working, roll it out to everyone
    - ⏳ **Extend the experiment** — promising signal but need more data to be sure  
    - ❌ **Do not ship** — no evidence the feature is helping, or it may be hurting
    """)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    st.divider()
    st.markdown("Built by **Rena Shrestha**")
    st.markdown("UC Berkeley Data Science '28")
    st.markdown("[GitHub](https://github.com/renashre/ai-feature-analyzer) · [LinkedIn](https://linkedin.com/in/renashrestha)")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# ── Run analysis ──────────────────────────────────────────────────────────────
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

    # ── FIX 1: Traffic light verdict ─────────────────────────────────────────
    st.subheader("📋 Verdict")

    if "Ship to all users" in verdict:
        st.success("✅ RECOMMENDATION: Ship to all users")
    elif "Extend the experiment" in verdict:
        st.warning("⏳ RECOMMENDATION: Extend the experiment before shipping")
    else:
        st.error("❌ RECOMMENDATION: Do not ship — insufficient evidence of lift")

    with st.expander("View full verdict details"):
        st.code(verdict, language=None)

    st.divider()

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        lift = stats["treatment_mean"] - stats["control_mean"]
        st.metric(
            label="⏱ Time Spent Per Visit",
            value=f"{stats['treatment_mean']:.1f}s",
            delta=f"{lift:.2f}s vs control"
        )
        st.caption("Higher = users spent more time after the AI feature launched")

    with col2:
        treatment_conv = engagement[engagement["group"] == "treatment"]["conversion_rate"].values[0]
        control_conv   = engagement[engagement["group"] == "control"]["conversion_rate"].values[0]
        st.metric(
            label="🛒 Purchase Rate",
            value=f"{treatment_conv:.2%}",
            delta=f"{treatment_conv - control_conv:.2%} vs control"
        )
        st.caption("% of sessions that resulted in a purchase")

    with col3:
        treatment_quality = user_quality[user_quality["group"] == "treatment"]["avg_user_quality_score"].values[0]
        control_quality   = user_quality[user_quality["group"] == "control"]["avg_user_quality_score"].values[0]
        st.metric(
            label="⭐ Engagement Quality",
            value=f"{treatment_quality:.0f}",
            delta=f"{treatment_quality - control_quality:.0f} vs control"
        )
        st.caption("Higher = users engaged more deeply, not just clicking around")

    with col4:
        p_value = stats["p_value"]
        st.metric(
            label="📐 Statistical Confidence",
            value=f"p = {p_value:.4f}",
            delta="significant ✓" if p_value < 0.05 else "not yet significant",
            delta_color="normal" if p_value < 0.05 else "inverse"
        )
        st.caption("p < 0.05 means the result is unlikely to be random chance")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # FIX 3: Retention chart — cut at week 28 and add note
    with col_left:
        st.subheader("📈 Retention Curves")
        st.caption("Are users coming back after the feature launched?")

        # cut at week 28 to avoid the data-edge drop
        retention_trimmed = retention[retention["weeks_since_launch"] <= 28]

        fig, ax = plt.subplots(figsize=(6, 4))
        for group, color in [("control", "#888780"), ("treatment", "#185FA5")]:
            group_data = retention_trimmed[retention_trimmed["group"] == group]
            ax.plot(
                group_data["weeks_since_launch"],
                group_data["retention_rate"],
                label=group.capitalize(),
                color=color,
                linewidth=2
            )
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Weeks Since Launch")
        ax.set_ylabel("Retention Rate (vs Week 1)")
        ax.set_title("Retention: Control vs Treatment")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.caption("📌 Chart shows weeks 0–28. A line above 1.0 means more users returned that week vs launch week.")

    # Novelty effect
    with col_right:
        st.subheader("🔍 Novelty Effect")
        st.caption("Is the engagement lift fading over time — or holding steady?")

        fig, ax = plt.subplots(figsize=(6, 4))
        novelty_trimmed = novelty[novelty["weeks_since_launch"] <= 28]
        ax.plot(
            novelty_trimmed["weeks_since_launch"],
            novelty_trimmed["treatment_lift"],
            color="#185FA5",
            linewidth=2,
            label="Treatment lift vs control"
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.fill_between(
            novelty_trimmed["weeks_since_launch"],
            novelty_trimmed["treatment_lift"],
            0,
            where=novelty_trimmed["treatment_lift"] > 0,
            alpha=0.1,
            color="#185FA5"
        )
        ax.fill_between(
            novelty_trimmed["weeks_since_launch"],
            novelty_trimmed["treatment_lift"],
            0,
            where=novelty_trimmed["treatment_lift"] < 0,
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
        st.caption("📌 Blue = treatment users more engaged than control. Red = control users more engaged. Consistently blue = real effect. Mixed = novelty or noise.")

    st.divider()

    # ── Engagement bar charts ─────────────────────────────────────────────────
    st.subheader("⚡ Engagement Breakdown")
    st.caption("Comparing how control and treatment users behaved after the feature launched")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        groups = engagement["group"].str.capitalize()
        values = engagement["avg_session_duration"]
        bars = ax.bar(groups, values, color=["#888780", "#185FA5"], width=0.4)
        ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=10)
        ax.set_ylabel("Avg Time Per Visit (seconds)")
        ax.set_title("Time Spent Per Visit")
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
        ax.set_title("Pages Explored Per Visit")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption("Built by Rena Shrestha · UC Berkeley Data Science '28 · ai-feature-analyzer.streamlit.app")

else:
    st.info("👈 Select an AI feature in the sidebar and click **Run Analysis** to get started.")
    st.markdown("""
    ### What can I evaluate?
    Select any of these four AI features from the sidebar:
    - 🤖 **AI Comment Suggestions** — does AI-suggested replies increase engagement?
    - 📸 **AI Generated Captions** — does auto-captioning change how users share content?
    - 🔍 **AI Search Assistant** — does AI search keep users on the platform longer?
    - 🎯 **AI Recommendations** — does AI-curated content improve retention?
    """)