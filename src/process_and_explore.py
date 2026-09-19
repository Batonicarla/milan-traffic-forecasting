import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["square_id", "time_interval", "country_code",
        "sms_in", "sms_out", "call_in", "call_out", "internet"]
USE_COLS = ["square_id", "time_interval", "internet"]
READ_DTYPES = {"square_id": "int32", "time_interval": "int64", "internet": "float32"}

txt_files = sorted(RAW_DIR.glob("*.txt"))
print(f"Found {len(txt_files)} raw files")


naive = pd.read_csv(txt_files[0], sep="\t", header=None, names=COLS)
before_mb = naive.memory_usage(deep=True).sum() / 1e6
print(f"Naive load of ONE day, default dtypes: {before_mb:.1f} MB | shape={naive.shape}")
del naive

def process_file_to_parquet(txt_path, out_path, chunksize=2_000_000):
    partials = []
    for chunk in pd.read_csv(txt_path, sep="\t", header=None, names=COLS,
                              usecols=USE_COLS, dtype=READ_DTYPES, chunksize=chunksize):
        chunk["internet"] = chunk["internet"].fillna(0)
        agg = chunk.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
        partials.append(agg)
    day_df = pd.concat(partials, ignore_index=True)
    day_df = day_df.groupby(["square_id", "time_interval"], as_index=False)["internet"].sum()
    day_df["internet"] = day_df["internet"].astype("float32")
    day_df.to_parquet(out_path, index=False)

t0 = time.time()
for f in txt_files:
    out_path = PROCESSED_DIR / (f.stem + ".parquet")
    if not out_path.exists():
        process_file_to_parquet(f, out_path)
print(f"Processed {len(txt_files)} files in {time.time() - t0:.1f}s")

parts = [pd.read_parquet(p) for p in sorted(PROCESSED_DIR.glob("*.parquet"))]
df = pd.concat(parts, ignore_index=True)

after_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"Full 62-day dataset in memory: {after_mb:.1f} MB | shape={df.shape}")

df["datetime"] = pd.to_datetime(df["time_interval"], unit="ms")
df = df.sort_values(["square_id", "datetime"]).reset_index(drop=True)
df.to_parquet(PROCESSED_DIR / "milan_internet_traffic_full.parquet", index=False)


total_by_square = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(total_by_square.values, bins=60)
ax.set_xlabel("Total Internet traffic over 2 months")
ax.set_ylabel("Number of geographical areas")
ax.set_title("Distribution of total Internet traffic across Milan's 10,000 squares")
plt.savefig("fig_traffic_distribution.png", dpi=150, bbox_inches="tight")
print(total_by_square.describe())


top3 = total_by_square.head(3).index.tolist()
target_squares = top3 + [4159, 4556]
print("Top 3 areas by total traffic:", top3)


start, end = "2013-11-01", "2013-11-15"
fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
for ax, sq in zip(axes, target_squares):
    sub = df[(df.square_id == sq) & (df.datetime >= start) & (df.datetime < end)]
    ax.plot(sub.datetime, sub.internet)
    label = "top traffic" if sq in top3 else "assigned square"
    ax.set_title(f"Square {sq} ({label})")
plt.tight_layout()
plt.savefig("fig_first_two_weeks.png", dpi=150, bbox_inches="tight")

print("Done. Check the two PNGs saved in your project folder.")