from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
df = pd.read_parquet(PROCESSED_DIR / "milan_internet_traffic_full.parquet")

total_by_square = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)
top3 = total_by_square.head(3).index.tolist()
print("Checking test week data quality for top-3 squares:", top3)

TEST_START, TEST_END = "2013-12-16", "2013-12-23"

for square_id in top3:
    ts = df[df.square_id == square_id].set_index("datetime")["internet"].asfreq("10min")
    test_week = ts[TEST_START:TEST_END]
    n_missing = test_week.isna().sum()
    print(f"\nSquare {square_id}: {n_missing} missing values in the test week")
    print(test_week.describe())