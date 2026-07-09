#create a function for each 5 questions 


#1. Engagement
#2. Retention
#3. User quality
#4. Statistical significance
#5. Novelty effect

#engagment 
from scipy import stats
import pandas as pd
import numpy as np

#1. Engagement
def calculate_engagement(df):
    post_df = df[df["period"] == "post"]
    
    engagement_summary = post_df.groupby("group").agg(
        total_sessions=("user_id", "count"),
        avg_session_duration=("session_duration_seconds", "mean"),
        avg_pages_visited=("pages_visited", "mean"),
        conversion_rate=("converted", "mean")
    ).reset_index()

    return engagement_summary


#2. Retention
def calculate_retention(df):
    post_df = df[df["period"] == "post"]

    retention_summary = post_df.groupby(["group", "weeks_since_launch"]).agg(
        unique_users=("user_id", "nunique")
    ).reset_index()

    week_1_counts = retention_summary[retention_summary["weeks_since_launch"] == 1].set_index("group")["unique_users"]

    retention_summary["retention_rate"] = retention_summary.apply(
        lambda row: row["unique_users"] / week_1_counts[row["group"]] if week_1_counts[row["group"]] > 0 else 0,
        axis=1
    )

    return retention_summary


#3. User quality
def calculate_user_quality(df):
    post_df = df[df["period"] == "post"].copy()

    post_df["user_quality_score"] = (
        post_df["session_duration_seconds"] * post_df["pages_visited"]
    )

    user_quality_summary = post_df.groupby("group").agg(
        avg_user_quality_score=("user_quality_score", "mean"),
        high_quality_users=("user_quality_score", lambda x: (x > 0.8).sum())
    ).reset_index()

    return user_quality_summary


#4. Statistical significance
def calculate_statistical_significance(df):
    post_df = df[df["period"] == "post"]

    control = post_df[post_df["group"] == "control"]["session_duration_seconds"]
    treatment = post_df[post_df["group"] == "treatment"]["session_duration_seconds"]

    t_stat, p_value = stats.ttest_ind(control, treatment)

    control_mean = control.mean()
    treatment_mean = treatment.mean()
    effect_size = (treatment_mean - control_mean) / control.std()

    ci = stats.t.interval(
        0.95,
        df=len(control) + len(treatment) - 2,
        loc=treatment_mean - control_mean,
        scale=stats.sem(treatment)
    )

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "effect_size": round(effect_size, 4),
        "confidence_interval": (round(ci[0], 4), round(ci[1], 4)),
        "control_mean": round(control_mean, 2),
        "treatment_mean": round(treatment_mean, 2)
    }


#5. Novelty effect
def calculate_novelty_effect(df):
    post_df = df[df["period"] == "post"]

    weekly_summary = post_df.groupby(["group", "weeks_since_launch"]).agg(
        avg_session_duration=("session_duration_seconds", "mean")
    ).reset_index()

    control_weekly = weekly_summary[weekly_summary["group"] == "control"][["weeks_since_launch", "avg_session_duration"]].set_index("weeks_since_launch")
    treatment_weekly = weekly_summary[weekly_summary["group"] == "treatment"][["weeks_since_launch", "avg_session_duration"]].set_index("weeks_since_launch")

    novelty_df = pd.DataFrame({
        "control_avg": control_weekly["avg_session_duration"],
        "treatment_avg": treatment_weekly["avg_session_duration"]
    }).dropna()

    novelty_df["treatment_lift"] = novelty_df["treatment_avg"] - novelty_df["control_avg"]
    novelty_df["novelty_flag"] = novelty_df["treatment_lift"] < novelty_df["treatment_lift"].shift(1)

    return novelty_df.reset_index()


if __name__ == "__main__":
    from data_loader import load_data
    from experiment import simulate_experiment

    df = load_data()
    df_exp = simulate_experiment(df, feature_name="AI Comment Suggestions")

    print("\n--- Engagement ---")
    print(calculate_engagement(df_exp))

    print("\n--- Retention ---")
    print(calculate_retention(df_exp))

    print("\n--- User Quality ---")
    print(calculate_user_quality(df_exp))

    print("\n--- Statistical Significance ---")
    print(calculate_statistical_significance(df_exp))

    print("\n--- Novelty Effect ---")
    print(calculate_novelty_effect(df_exp))

                                                            