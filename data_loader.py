import pandas as pd

def load_data(filepath="ga_sessions.csv"):
    df = pd.read_csv(filepath, low_memory=False)

    df["session_duration_seconds"] = df["session_duration_seconds"].fillna(0)
    df["pages_visited"] = df["pages_visited"].fillna(0)
    df["converted"] = df["converted"].fillna(0).astype(int)


    df["session_date"] = pd.to_datetime(df["session_date"])

    df = df.dropna(subset=["user_id"])

    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Rows: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")