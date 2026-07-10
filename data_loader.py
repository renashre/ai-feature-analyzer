import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

QUERY = """
SELECT
  fullVisitorId                           AS user_id,
  DATE(TIMESTAMP_SECONDS(visitStartTime)) AS session_date,
  totals.timeOnSite                       AS session_duration_seconds,
  totals.pageviews                        AS pages_visited,
  totals.transactions                     AS converted,
  device.deviceCategory                   AS device,
  channelGrouping                         AS channel,
  geoNetwork.country                      AS country
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  totals.visits = 1
  AND _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
"""

@st.cache_data
def load_data():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(
        credentials=credentials,
        project=st.secrets["gcp_service_account"]["project_id"]
    )
    df = client.query(QUERY).to_dataframe()

    df["session_duration_seconds"] = df["session_duration_seconds"].fillna(0)
    df["pages_visited"] = df["pages_visited"].fillna(0)
    df["converted"] = df["converted"].fillna(0).astype(int)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.dropna(subset=["user_id"])

    return df