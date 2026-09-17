import pandas as pd

df = pd.read_parquet("data/processed/milan_internet_traffic_full.parquet")
ts = df[df.square_id == 5161].set_index("datetime")["internet"].asfreq("10min")

test_week = ts["2013-12-16":"2013-12-23"]
print(test_week.isna().sum(), "missing values in the test week")
print(test_week.describe())