def generate_verdict(feature_name, engagement, retention, user_quality, stats, novelty):

    # 1. Engagement
    lift = stats["treatment_mean"] - stats["control_mean"]
    if lift > 0:
        eng_text = f"The treatment group had a higher average session duration by {lift:.2f}s ({stats['treatment_mean']}s vs {stats['control_mean']}s for control)."
    else:
        eng_text = f"No engagement lift detected. Treatment group session duration was {stats['treatment_mean']}s vs {stats['control_mean']}s for control."

    treatment_conversion = engagement[engagement["group"] == "treatment"]["conversion_rate"].values[0]
    control_conversion = engagement[engagement["group"] == "control"]["conversion_rate"].values[0]
    conv_text = f"Conversion rate: treatment {treatment_conversion:.2%} vs control {control_conversion:.2%}."

    # 2. Retention
    final_week_retention = retention[retention["group"] == "treatment"]["retention_rate"].iloc[-1]
    if final_week_retention > 0.8:
        ret_text = f"Retention held strong at {final_week_retention:.0%} by the final week."
    else:
        ret_text = f"Retention dropped to {final_week_retention:.0%} by the final week — worth monitoring."

    # 3. User quality
    treatment_quality = user_quality[user_quality["group"] == "treatment"]["avg_user_quality_score"].values[0]
    control_quality = user_quality[user_quality["group"] == "control"]["avg_user_quality_score"].values[0]
    if treatment_quality >= control_quality:
        quality_text = f"User quality improved or held steady (treatment {treatment_quality:.0f} vs control {control_quality:.0f})."
    else:
        quality_text = f"User quality dropped (treatment {treatment_quality:.0f} vs control {control_quality:.0f}) — flag for review."

    # 4. Statistical significance
    p_value = stats["p_value"]
    effect_size = stats["effect_size"]
    ci = stats["confidence_interval"]
    if p_value < 0.05:
        sig_text = f"The lift is statistically significant (p={p_value}, effect size={effect_size}, 95% CI=({ci[0]:.4f}, {ci[1]:.4f}))."
    elif p_value < 0.1:
        sig_text = f"The lift is directional but not yet statistically significant (p={p_value}, effect size={effect_size}, 95% CI=({ci[0]:.4f}, {ci[1]:.4f})). More data needed."
    else:
        sig_text = f"No statistically significant lift detected (p={p_value}, effect size={effect_size}, 95% CI=({ci[0]:.4f}, {ci[1]:.4f}))."

    # 5. Novelty effect
    flagged_weeks = novelty["novelty_flag"].sum()
    total_weeks = len(novelty)
    if flagged_weeks > total_weeks * 0.5:
        novelty_text = f"Lift is declining over time ({flagged_weeks}/{total_weeks} weeks flagged) — likely a novelty effect."
    else:
        novelty_text = f"No consistent novelty effect detected ({flagged_weeks}/{total_weeks} weeks flagged) — lift appears stable."

    # 6. Recommendation
    if p_value < 0.05 and treatment_quality >= control_quality:
        recommendation = "RECOMMENDATION: Ship to all users."
    elif p_value < 0.1:
        recommendation = "RECOMMENDATION: Extend the experiment before shipping."
    else:
        recommendation = "RECOMMENDATION: Do not ship — insufficient evidence of lift."

    # 7. Assemble verdict
    verdict = f"""
====================================
AI FEATURE IMPACT ANALYZER
====================================
Feature: {feature_name}

ENGAGEMENT
{eng_text}
{conv_text}

RETENTION
{ret_text}

USER QUALITY
{quality_text}

STATISTICAL SIGNIFICANCE
{sig_text}

NOVELTY EFFECT
{novelty_text}

------------------------------------
{recommendation}
------------------------------------
"""

    return verdict


if __name__ == "__main__":
    from data_loader import load_data
    from experiment import simulate_experiment
    from analysis import (
        calculate_engagement,
        calculate_retention,
        calculate_user_quality,
        calculate_statistical_significance,
        calculate_novelty_effect
    )

    df = load_data()
    df_exp = simulate_experiment(df, feature_name="AI Comment Suggestions")

    engagement = calculate_engagement(df_exp)
    retention = calculate_retention(df_exp)
    user_quality = calculate_user_quality(df_exp)
    stats = calculate_statistical_significance(df_exp)
    novelty = calculate_novelty_effect(df_exp)

    verdict = generate_verdict(
        feature_name="AI Comment Suggestions",
        engagement=engagement,
        retention=retention,
        user_quality=user_quality,
        stats=stats,
        novelty=novelty
    )

    print(verdict)