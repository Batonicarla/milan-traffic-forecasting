# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

A comparative study of SARIMA, LSTM, and GRU for one-step-ahead mobile Internet
traffic forecasting, using the Milan Telecom Italia dataset (telecommunications
activity across 10,000 geographical areas in Milan, Nov 2013 - Jan 2014).

## Project structure
milan-traffic-forecasting/
        ├── data/ (not included in this repo — see setup below)
        ├── raw/ 62 daily .txt files from Harvard Dataverse
        ├── grid/ Milano Grid geo-boundary files (downloaded, not used in analysis)
         └── processed/ memory-optimized Parquet files built by src/process_and_explore.py
─ src/
     ├── download_data.py downloads the raw dataset from Harvard Dataverse
     ├── process_and_explore.py memory-efficient processing + core exploratory analysis
     ├── extra_analysis.py seasonal decomposition, ACF/PACF, stationarity test
     ├── check_test_week.py data quality check on the evaluation week
    └── modeling.py trains/evaluates SARIMA, LSTM, GRU across the top-3 traffic areas
 results/
      ├── figures/ all generated plots (EDA + 9 prediction comparison plots)
      └── tables/ MAE/MAPE/RMSE results per area + training/inference timing
├── requirements.txt
└── README.md


## Setup

### 1. Install dependencies
Requires Python 3.10+.
```bash
pip install -r requirements.txt
```

### 2. Get a Harvard Dataverse account and API token
The dataset requires a one-time "guestbook" response (name/institution/purpose)
tied to your account before any file can be downloaded, even via the API.

1. Create a free account at https://dataverse.harvard.edu
2. While logged in, go to the dataset page and click Download on any one file:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
   Fill in the guestbook pop-up that appears and click Accept — this satisfies
   the requirement for your account.
3. Generate an API token at:
   https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab
4. Keep this token handy — you'll be prompted to paste it when running the
   download script (it is never stored in this repo).

### 3. Download the dataset
```bash
python src/download_data.py
```
Paste your API token when prompted. This downloads 62 daily files (~15-20 GB
total) plus the Milano Grid geo-reference files. Safe to re-run — it skips any
file already downloaded.

## Running the pipeline

Run each script from the project root, in this order:

```bash
python src/process_and_explore.py
python src/extra_analysis.py
python src/check_test_week.py
python src/modeling.py
```

- **`process_and_explore.py`**  converts the 62 raw files into a compact,
  memory-efficient Parquet dataset, and produces the traffic distribution and
  first-two-weeks comparison figures. Reports memory usage before/after
  optimization.
- **`extra_analysis.py`**  STL seasonal decomposition (daily and weekly) and
  ACF/PACF/ADF stationarity analysis on the highest-traffic area.
- **`check_test_week.py`**  verifies that there is no missing data in the
  evaluation window (Dec 16-22, 2013).
- **`modeling.py`**  the main experiment. Trains SARIMA, LSTM, and GRU for
  one-step-ahead forecasting on each of the three highest-traffic areas,
  evaluates on Dec 16-22, and saves results tables, timing statistics, and
  9 comparison plots (3 models x 3 areas).

**Note on runtime:** `modeling.py` is computationally heavy — SARIMA's seasonal
fit plus hyperparameter searches for LSTM and GRU across 3 areas can take
45 minutes to a few hours on a CPU-only machine, with no GPU acceleration
(This repo was developed and run on a Windows laptop with no GPU.) This is
expected; the script does not hang.

## Outputs

After running the full pipeline, `results/figures/` contains:
- `fig_traffic_distribution.png`, `fig_first_two_weeks.png` — exploratory analysis
- `fig_stl_daily.png`, `fig_stl_weekly.png`, `fig_acf_pacf.png` — seasonal/autocorrelation analysis
- `fig_pred_<square_id>_<model>.png` (9 files) — actual vs. predicted traffic per model per area

`results/tables/` contains:
- `results_square_<square_id>.csv` (3 files) — MAE/MAPE/RMSE per model per area
- `timing_stats.csv` — training and inference time per model

## Models

Three models were selected to give a meaningful comparison between a classical
statistical approach and modern sequence models, justified by exploratory
evidence of strong daily/weekly seasonality (see `results/figures/fig_acf_pacf.png`
and the STL decomposition):

- **SARIMA**  explicit seasonal modeling, used as a classical baseline
- **LSTM**  recurrent neural network, gated memory
- **GRU**  recurrent neural network, simplified gating

Input sequence length (144 steps = 1 day of 10-minute intervals) was chosen
based on the ACF analysis showing the strongest autocorrelation at the daily
lag. SARIMA orders were selected from the EDA evidence rather than an
exhaustive grid search, since a seasonal grid search at this seasonal period
(m=144) is computationally prohibitive; this trade-off is discussed in the
accompanying report.

## AI usage disclosure

AI assistance (Claude) was used for code scaffolding, debugging environment
and dependency issues, and structuring this repository. All modeling
decisions, interpretation of results, and the written report reflect the
author's own understanding and analysis.

## Author
 (Carla Batoni)
