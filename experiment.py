#1. Split users into two groups
#2. Pick a launch date and tag sessions
#3. Calculate weeks since launch

import pandas as pd
import numpy as np

FEATURES = [
    "AI Comment Suggestions",
    "AI Generated Captions",
    "AI Search Assistant",
    "AI Recommendations"
]

def simulate_experiment(df, feature_name, launch_date="2017-01-01", seed=42):
    np.random.seed(seed)

    # Feature-specific effect sizes (realistic but varied)
    feature_effects = {
        "AI Comment Suggestions": {
            "session_boost": 0.08,
            "conversion_boost": 0.002,
            "quality_boost": 0.10
        },
        "AI Generated Captions": {
            "session_boost": 0.15,
            "conversion_boost": 0.005,
            "quality_boost": 0.20
        },
        "AI Search Assistant": {
            "session_boost": 0.25,
            "conversion_boost": 0.010,
            "quality_boost": 0.30
        },
        "AI Recommendations": {
            "session_boost": -0.05,
            "conversion_boost": -0.001,
            "quality_boost": -0.08
        }
    }

    effects = feature_effects.get(feature_name, {
        "session_boost": 0.05,
        "conversion_boost": 0.001,
        "quality_boost": 0.05
    })

    unique_users = df["user_id"].unique()
    treatment_users = set(
        np.random.choice(unique_users, size=len(unique_users) // 2, replace=False)
    )

    df = df.copy()
    df["feature"] = feature_name
    df["group"] = df["user_id"].apply(
        lambda uid: "treatment" if uid in treatment_users else "control"
    )

    launch_date = pd.Timestamp(launch_date)
    df["period"] = df["session_date"].apply(
        lambda d: "post" if d >= launch_date else "pre"
    )

    df["weeks_since_launch"] = (
        (df["session_date"] - launch_date).dt.days // 7
    )

    # Inject realistic signal for treatment group post-launch
    post_treatment = (df["group"] == "treatment") & (df["period"] == "post")

    df.loc[post_treatment, "session_duration_seconds"] = (
        df.loc[post_treatment, "session_duration_seconds"] *
        (1 + effects["session_boost"] + np.random.normal(0, 0.02, post_treatment.sum()))
    ).clip(lower=0)

    df.loc[post_treatment, "converted"] = np.where(
        np.random.random(post_treatment.sum()) 
        (df.loc[post_treatment, "converted"].mean() + effects["conversion_boost"]),
        1, df.loc[post_treatment, "converted"]
    )

    return df


def get_experiment_summary(df):
    """
    Prints a quick sanity check of the experiment setup
    """
    print(f"\nFeature: {df['feature'].iloc[0]}")
    print(f"Total sessions: {len(df)}")
    print(f"\nGroup split:")
    print(df["group"].value_counts())
    print(f"\nPeriod split:")
    print(df["period"].value_counts())
    print(f"\nSample of experiment data:")
    print(df[["user_id", "session_date", "group", "period", "weeks_since_launch"]].head(10))


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df_exp = simulate_experiment(df, feature_name="AI Comment Suggestions")
    get_experiment_summary(df_exp)

