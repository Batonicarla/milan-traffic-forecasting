import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)

PROCESSED_DIR = Path("data/processed")
df = pd.read_parquet(PROCESSED_DIR / "milan_internet_traffic_full.parquet")

total_by_square = df.groupby("square_id")["internet"].sum().sort_values(ascending=False)
top3 = total_by_square.head(3).index.tolist()
print("Top 3 areas:", top3)

SEQ_LEN = 144
TEST_START = "2013-12-16"
TEST_END = "2013-12-23"


def get_square_series(square_id):
    return df[df.square_id == square_id].set_index("datetime")["internet"].asfreq("10min").interpolate()


def train_test_split_series(s):
    train = s[s.index < TEST_START]
    test = s[(s.index >= TEST_START) & (s.index < TEST_END)]
    return train, test


def make_windows(series_values, seq_len):
    X, y = [], []
    for i in range(len(series_values) - seq_len):
        X.append(series_values[i:i + seq_len])
        y.append(series_values[i + seq_len])
    return np.array(X), np.array(y)


def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}


def run_sarima(train, test, order=(1, 0, 1), seasonal_order=(0, 1, 1, 144)):
    t0 = time.time()
    model = SARIMAX(train.values, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=50, low_memory=True)
    train_time = time.time() - t0

    t0 = time.time()
    combined = np.concatenate([train.values, test.values])
    full_model = SARIMAX(combined, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
    full_res = full_model.filter(fit.params)
    preds = full_res.predict(start=len(train), end=len(combined) - 1)
    infer_time = time.time() - t0
    return np.array(preds), train_time, infer_time


def build_model(cell_type, units, lr, seq_len):
    layer = keras.layers.LSTM if cell_type == "lstm" else keras.layers.GRU
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, 1)),
        layer(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def tune_and_train(cell_type, X_train, y_train, X_val, y_val, seq_len):
    grid = [
        {"units": 32, "lr": 1e-3},
        {"units": 64, "lr": 1e-3},
        {"units": 64, "lr": 5e-4},
    ]
    results = []
    for cfg in grid:
        model = build_model(cell_type, cfg["units"], cfg["lr"], seq_len)
        es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=20, batch_size=256, callbacks=[es], verbose=0)
        val_rmse = min(hist.history["val_loss"]) ** 0.5
        results.append({**cfg, "val_rmse": val_rmse, "model": model})
        print(f"  [{cell_type}] units={cfg['units']} lr={cfg['lr']}  val_RMSE={val_rmse:.3f}")

    best = min(results, key=lambda r: r["val_rmse"])
    print(f"  [{cell_type}] best config: units={best['units']} lr={best['lr']}")
    return best["model"], best


all_results = {}
timing_stats = {}
predictions_store = {}

for idx, square_id in enumerate(top3):
    print(f"\n=== Square {square_id} ===")
    s = get_square_series(square_id)
    train_series, test_series = train_test_split_series(s)

    scaler = MinMaxScaler()
    scaler.fit(train_series.values.reshape(-1, 1))
    combined_scaled = scaler.transform(
        pd.concat([train_series, test_series]).values.reshape(-1, 1)
    ).flatten()
    n_train = len(train_series)

    X_all, y_all = make_windows(combined_scaled, SEQ_LEN)
    split_point = n_train - SEQ_LEN
    val_cut = int(split_point * 0.9)
    X_train, y_train = X_all[:val_cut], y_all[:val_cut]
    X_val, y_val = X_all[val_cut:split_point], y_all[val_cut:split_point]
    X_test = X_all[split_point:]

    X_train_r = X_train.reshape(-1, SEQ_LEN, 1)
    X_val_r = X_val.reshape(-1, SEQ_LEN, 1)
    X_test_r = X_test.reshape(-1, SEQ_LEN, 1)

    square_results = {}
    square_preds = {}

    print("Training SARIMA...")
    sarima_preds, sarima_train_t, sarima_infer_t = run_sarima(train_series, test_series)
    square_results["SARIMA"] = evaluate_metrics(test_series.values, sarima_preds)
    square_preds["SARIMA"] = sarima_preds
    if idx == 0:
        timing_stats["SARIMA"] = {"train_s": sarima_train_t, "inference_s": sarima_infer_t}

    print("Training LSTM (hyperparameter search)...")
    t0 = time.time()
    lstm_model, lstm_cfg = tune_and_train("lstm", X_train_r, y_train, X_val_r, y_val, SEQ_LEN)
    lstm_train_t = time.time() - t0
    t0 = time.time()
    lstm_pred = scaler.inverse_transform(
        lstm_model.predict(X_test_r, verbose=0).reshape(-1, 1)
    ).flatten()
    lstm_infer_t = time.time() - t0
    square_results["LSTM"] = evaluate_metrics(test_series.values, lstm_pred)
    square_preds["LSTM"] = lstm_pred
    if idx == 0:
        timing_stats["LSTM"] = {"train_s": lstm_train_t, "inference_s": lstm_infer_t, "config": lstm_cfg}

    print("Training GRU (hyperparameter search)...")
    t0 = time.time()
    gru_model, gru_cfg = tune_and_train("gru", X_train_r, y_train, X_val_r, y_val, SEQ_LEN)
    gru_train_t = time.time() - t0
    t0 = time.time()
    gru_pred = scaler.inverse_transform(
        gru_model.predict(X_test_r, verbose=0).reshape(-1, 1)
    ).flatten()
    gru_infer_t = time.time() - t0
    square_results["GRU"] = evaluate_metrics(test_series.values, gru_pred)
    square_preds["GRU"] = gru_pred
    if idx == 0:
        timing_stats["GRU"] = {"train_s": gru_train_t, "inference_s": gru_infer_t, "config": gru_cfg}

    all_results[square_id] = square_results
    predictions_store[square_id] = {"test_series": test_series, "preds": square_preds}

print("\nAll experiments complete.\n")


for square_id, results in all_results.items():
    label = "TOP TRAFFIC AREA" if square_id == top3[0] else f"area rank #{top3.index(square_id)+1}"
    print(f"\n--- Square {square_id} ({label}) ---")
    print(pd.DataFrame(results).T.round(3))

print("\n--- Training/inference time (measured on the top-traffic area) ---")
print(pd.DataFrame(timing_stats).T)


for square_id, results in all_results.items():
    pd.DataFrame(results).T.round(3).to_csv(f"results_square_{square_id}.csv")
pd.DataFrame(timing_stats).T.to_csv("timing_stats.csv")


for square_id in top3:
    test_series = predictions_store[square_id]["test_series"]
    for model_name, pred in predictions_store[square_id]["preds"].items():
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(test_series.index, test_series.values, label="Actual", linewidth=1.5)
        ax.plot(test_series.index, pred, label=f"Predicted ({model_name})", linewidth=1.2)
        ax.set_title(f"Square {square_id} - {model_name} - Dec 16-22")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"fig_pred_{square_id}_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

print("\nDone. Check the CSV tables and the 9 fig_pred_*.png files.")