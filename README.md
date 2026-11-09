# Does News Move Kalshi, and How Fast?

CS 584 project — Peter Reid, Randeep Chahal, Pranav Ramchandra Auti (Stevens Institute of Technology)

This repo holds the data pipeline for testing whether news coverage, and the stance of that coverage toward a contract's outcome, predicts short-horizon price changes on Kalshi prediction markets. The pilot covers 21 contracts traded during 2025.

## Pipeline

The notebook `kalshi_data_collection.ipynb` runs each stage in order, importing the modules below.

| Stage | Module | Output |
|---|---|---|
| 1. Pull 2025 Kalshi markets and screen candidates | notebook cells | `screening_df` |
| 2. Select the pilot sample (settled, ≥14 days active in 2025, liquid, one contract per event, ≤2 per series) | `kalshi_revised_sample.py` | `kalshi_pilot_revised_contracts.csv` |
| 3. Hourly 2025 price histories from Kalshi candlesticks | `kalshi_price_history.py` | `data/pilot/pilot_prices_hourly_2025.csv` |
| 4. Match news headlines to contracts via GDELT in BigQuery | `kalshi_gdelt_bigquery.py` | `data/pilot/pilot_articles.csv` |
| 5. Relevance filter (sentence-transformer), VADER and FinBERT sentiment, stance annotation sample | `kalshi_news_scoring.py` | `data/pilot/pilot_articles_scored.csv`, `stance_annotation_sample.csv` |
| 6. Leakage-safe modeling table and association analysis | `kalshi_model_dataset.py` | `data/pilot/model_dataset.csv` |

`kalshi_sample.py` and `kalshi_gdelt_news.py` are earlier versions (first-pass sample selection and the GDELT DOC API collector). They are kept for reference; the DOC API proved too slow and rate-limited for a full year of coverage, so collection moved to BigQuery.

## Data

Data files are not in the repo (several exceed GitHub's size limit). Download the shared data folder here: **[https://drive.google.com/drive/folders/1My0ZRJsHHYhKunsa_7DZPWfpx0RCjKgt?usp=drive_link]**

Place its contents so the paths match the table above (`data/pilot/...` and the two contract CSVs in the project root). With the data in place, the notebook can start from the "Building the modeling dataset" section without rerunning collection.

## Setup

```bash
pip install -r requirements.txt
```

Rerunning the BigQuery collection (stage 4) requires a Google Cloud project and authentication:

```bash
gcloud auth application-default login
```

Set `PROJECT` in the notebook to your own project ID. A full scan of 2025 uses about 77 GB, within BigQuery's free 1 TB per month. Raw matches are saved locally, so changing keyword terms re-filters without a new scan. The relevance and FinBERT models download from Hugging Face on first use.

## Modeling dataset

Each row of `model_dataset.csv` is one contract at one checkpoint (every 6 hours) for one pair of lookback window w (1, 3, 6, 12, 24, 72 h) and horizon h (1, 3, 6, 12, 24 h).

- **News features:** relevant article count, distinct stories, story-level mean VADER, FinBERT and GDELT tone over the w hours before the checkpoint; `net_stance` once stance labels are added.
- **Market features:** price, recent price change, volume and spread at the checkpoint.
- **Targets:** price change over the next h hours and direction (up / flat / down, with ±2¢ as flat).
- **Leakage control:** article times are rounded up to the hour, so an article counts only at checkpoints at or after its publication. `split_group` groups contracts that share news (e.g. the Fed contracts) for grouped train/test splits.

## Status

- **Done:** contract selection, price histories, article collection and scoring, modeling table, first association analysis.
- **Early findings:** article volume is consistently associated with larger subsequent price moves (Spearman 0.11–0.18 across window pairs, strongest near w = 3 h, h = 12 h). Headline sentiment shows no relationship with move direction (|ρ| < 0.03), consistent with the case for contract-specific stance. Correlations are pooled and from overlapping rows, so they are descriptive only.
- **Next:** stance annotation (two annotators, agreement reported), stance model, market-only baseline vs. baseline plus news features, clustered bootstrap confidence intervals.

## Known limitations

- Matching is on headlines only; full article text is not used.
- About 94% of targets are "flat", so evaluation should use macro-F1 and balanced accuracy.
- Several politics contracts share national news and are not fully independent.
- Climate and weather contracts were excluded: Kalshi's 2025 weather markets are short-lived, forecast-driven daily contracts with little news coverage.
