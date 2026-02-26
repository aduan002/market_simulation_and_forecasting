# README

Financial markets are constantly shaped by unpredictable future events, making short-horizon price dynamics appear close to noise. This project asks a fundamental question:

**If market participants behave randomly, does any predictable structure still emerge?**

To isolate this, I construct a synthetic limit order book populated entirely by stochastic agents. Orders are generated probabilistically, with no informational edge or strategic behavior. The environment therefore approximates a noise-dominated market driven purely by microstructure mechanics.

Despite this, both machine learning models and classical econometric time-series approaches detect statistically significant predictive structure. This suggests that structured dynamics can emerge endogenously from the mechanics of the order book itself.

A detailed performance analysis and statistical validation are available [here](misc\report.md).

## Project Structure
```bash
.
├── 00_simulate.py
├── 01_make_features.py
├── 02_train.py
├── 03_filter_log.py
├── 04_compute_statistical_analysis.py
├── agents/
├── cfg/
├── data/
├── datasets/
├── features/
├── logs/
├── metrics/
├── models/
├── orderbook/
├── runs/
├── statistical_significance_results/
├── utils/
├── README.md
└── requirements.txt
```

## Project Overview

### 1. Simulation
Simulates price evolution from a bounded order book populated by random agents.
- Price placement: Poisson distribution
- Order size: Negative binomial distribution
- Price evolution driven by microprice dynamics
- Second-level time resolution

#### Output
- CSV file with:
    - t (time step)
    - price
- Price plot visualization

#### Usage
```bash
python 00_simulate.py -c cfg/simulate_config.yaml -o data/simulated_prices.csv
```

### 2. Feature Engineering
Generates volatility and price based features from price data.

#### Output 
- CSV file with:
    - t
    - price
    - lagged log returns
    - lagged prices
    - lagged volatility
    - target (at next hour)

#### Usage:
```bash 
python 01_make_features.py -i data/simulated_prices.csv -o features/features.csv
```

### 3. Model Training & Hyperparameter Search
Trains multiple regression algorithms using:
- Feature and target preprocessing pipelines
- Time-series cross-validation
- Custom grid search 

#### Output
- Folder with:
    - Logged per-fold metrics.
    - Preprocessing objects per fold
    - Model objects per fold

#### Usage:
```bash
python 02_train.py -c cfg/workflow_config.yaml -p cfg/hp.json
```

### 4. Log Filtering
Parses log files to extract:
- Best hyperparameter configuration per algorithm
- Fold-wise metrics for the best configuration
- Clean CSV for statistical comparison

#### Output
- CSV file with:
    - algorithm name
    - fold number
    - error metrics

#### Usage
```bash
python 03_filter_log.py -i runs/your_run/log.txt -o metrics/best_models.csv
```

### 4. Statistical Significance & Ranking
Performs statistical comparison between algorithms:
- T-test
- Confidence Interval
- Ranking based on:
    - Wins given confidence intervals

#### Output
- CSV file with:
    - algorithm name
    - mean of metric
    - standard of metric
    - rank

#### Usage
```bash
python 04_compute_statistical_analysis.py -i metrics/best_models.csv -o statistical_significance_results/ranked_models.csv
```



## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```