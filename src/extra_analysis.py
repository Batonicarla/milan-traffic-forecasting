from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = Path("data/processed")

df = pd.read_parquet(PROCESSED_DIR / "milan_internet_traffic_full.parquet")

total_by_square = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)
top3 = total_by_square.head(3).index.tolist()
top_square = top3[0]
print(f"Top traffic square: {top_square}")

ts = df[df.square_id == top_square].set_index("datetime")["internet"].asfreq("10min").interpolate()


stl_daily = STL(ts, period=144, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
axes[0].plot(ts.index, ts.values); axes[0].set_ylabel("Observed")
axes[1].plot(ts.index, stl_daily.trend); axes[1].set_ylabel("Trend")
axes[2].plot(ts.index, stl_daily.seasonal); axes[2].set_ylabel("Daily seasonal")
axes[3].plot(ts.index, stl_daily.resid); axes[3].set_ylabel("Residual")
plt.suptitle(f"STL decomposition (daily period) - Square {top_square}")
plt.tight_layout()
plt.savefig("fig_stl_daily.png", dpi=150, bbox_inches="tight")
plt.close()

stl_weekly = STL(ts, period=144 * 7, robust=True).fit()
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ts.index, stl_weekly.seasonal)
ax.set_title(f"Weekly seasonal component - Square {top_square}")
plt.savefig("fig_stl_weekly.png", dpi=150, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(2, 1, figsize=(11, 7))
plot_acf(ts, lags=288, ax=axes[0]); axes[0].set_title(f"ACF - Square {top_square}")
plot_pacf(ts, lags=50, ax=axes[1], method="ywm"); axes[1].set_title(f"PACF - Square {top_square}")
plt.tight_layout()
plt.savefig("fig_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()

result = adfuller(ts.dropna())
print(f"ADF statistic: {result[0]:.3f}  p-value: {result[1]:.4f}")
print("Stationary" if result[1] < 0.05 else "Non-stationary - differencing needed for SARIMA")

# --- Spatial heatmap: total traffic across the 100x100 Milan grid ---
grid = np.zeros(10000)
grid[total_by_square.index.values - 1] = total_by_square.values
grid = grid.reshape(100, 100)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(grid, cmap="viridis", origin="lower")
ax.set_title("Total Internet traffic across Milan grid (Nov 2013 - Jan 2014)")
plt.colorbar(im, ax=ax, label="Total traffic")
plt.savefig("fig_spatial_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Weekday vs weekend average diurnal profile ---
ts_df = ts.reset_index()
ts_df["hour"] = ts_df["datetime"].dt.hour + ts_df["datetime"].dt.minute / 60
ts_df["is_weekend"] = ts_df["datetime"].dt.dayofweek >= 5

profile = ts_df.groupby(["is_weekend", "hour"])["internet"].mean().unstack(level=0)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(profile.index, profile[False], label="Weekday")
ax.plot(profile.index, profile[True], label="Weekend")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean internet traffic")
ax.set_title(f"Average diurnal profile - Square {top_square}")
ax.legend()
plt.savefig("fig_weekday_weekend.png", dpi=150, bbox_inches="tight")
plt.close()

print("Done. Check fig_stl_daily.png, fig_stl_weekly.png, fig_acf_pacf.png,")
print("fig_spatial_heatmap.png, and fig_weekday_weekend.png")