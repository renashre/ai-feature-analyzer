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
    """
    Simulates an AI feature launch experiment.
    - Splits users into control and treatment groups
    - Tags sessions as pre or post launch
    - Returns the full dataframe with experiment columns added
    """

    np.random.seed(seed)

    # get all unique users and randomly assign to control or treatment
    unique_users = df["user_id"].unique()
    treatment_users = set(
        np.random.choice(unique_users, size=len(unique_users) // 2, replace=False)
    )

    # tag each row
    df = df.copy()
    df["feature"] = feature_name
    df["group"] = df["user_id"].apply(
        lambda uid: "treatment" if uid in treatment_users else "control"
    )

    launch_date = pd.Timestamp(launch_date)
    df["period"] = df["session_date"].apply(
        lambda d: "post" if d >= launch_date else "pre"
    )

    # add week number relative to launch (useful for novelty effect detection later)
    df["weeks_since_launch"] = (
        (df["session_date"] - launch_date).dt.days // 7
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