# Equity Forecasting Architecture Documentation

**Last Updated:** March 2026  
**Project Status:** 3/10 institutional grade → Target: 8/10 (pre-AWS deployment)  
**Version:** 1.0.0

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture Diagram](#system-architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Pipeline Stages (A-H)](#pipeline-stages-a-h)
6. [Models Architecture](#models-architecture)
7. [Ensemble Strategy](#ensemble-strategy)
8. [API Layer](#api-layer)
9. [Dashboard Integration](#dashboard-integration)
10. [Configuration System](#configuration-system)
11. [Validation & Monitoring](#validation--monitoring)
12. [File Structure](#file-structure)
13. [Dependencies & Versioning](#dependencies--versioning)
14. [Design Patterns](#design-patterns)
15. [Current Status & Known Issues](#current-status--known-issues)
16. [Performance Considerations](#performance-considerations)

---

## Project Overview

**Equity Forecasting** is an end-to-end machine learning system for predicting equity price movements using:
- Multiple deep learning models (LSTM, TCN)
- Traditional ML models (XGBoost, LightGBM, Ridge, ElasticNet)
- Market sentiment analysis
- Technical indicators
- Ensemble meta-learning

**Key Objectives:**
- Predict future equity prices with high accuracy
- Provide multi-horizon forecasts (1-30 days ahead)
- Support multiple equities simultaneously
- Real-time sentiment analysis
- Interactive web dashboard for visualization
- REST API for programmatic access
- Production-ready with AWS deployment capability

**Target Users:**
- Traders and portfolio managers
- Quantitative analysts
- ML practitioners
- Financial institutions

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EQUITY FORECASTING SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   CLI Entry      │
                              │   (main.py)      │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            ┌──────────────┐    ┌────────────┐   ┌──────────────┐
            │  Orchestrator│    │ DAG Runner │   │  API Server  │
            │              │    │            │   │  (FastAPI)   │
            └──────┬───────┘    └────────┬───┘   └──────┬───────┘
                   │                     │              │
        ┌──────────┼─────────────────────┼──────────────┼─────────┐
        │          │                     │              │         │
        ▼          ▼                     ▼              ▼         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              PIPELINE STAGES (A → H)                            │
    ├─────────────────────────────────────────────────────────────────┤
    │ A: Ingestion   → B: Preprocess  → C: Features  → D: Optimize   │
    │ E: Model Train → F: Inference   → G: Walk-Fwd  → H: Ensemble   │
    └─────────────────────────────────────────────────────────────────┘
        │          │                     │              │         │
        ▼          ▼                     ▼              ▼         ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │            FEATURE GENERATION & SENTIMENT ANALYSIS               │
    ├──────────────────────────────────────────────────────────────────┤
    │  Technical Indicators  │  Market Sentiment  │  Global Signals   │
    │  (MA, RSI, MACD, etc)  │  (News, Twitter)   │  (Cross-equity)   │
    └──────────────────────────────────────────────────────────────────┘
        │                     │                          │
        ▼                     ▼                          ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                    MODEL ENSEMBLE                                │
    ├──────────────────────────────────────────────────────────────────┤
    │  LSTM      TCN       XGBoost   LightGBM   Ridge   ElasticNet    │
    │  (RNN)    (1D-CNN)   (Boost)   (Boost)   (Linear) (Linear)      │
    │                                                                  │
    │  Meta-Learner (Stacking):  LightGBM or Linear Layer             │
    └──────────────────────────────────────────────────────────────────┘
        │                                                   │
        ▼                                                   ▼
    ┌──────────────────────┐                    ┌──────────────────────┐
    │   Dashboard           │                    │   REST API           │
    │   (Streamlit)         │                    │   (FastAPI)          │
    │                       │                    │                      │
    │ - Forecast View       │                    │ /forecast (GET/POST) │
    │ - Sentiment Panel     │                    │ /sentiment (POST)    │
    │ - History Manager     │                    │ /train (POST)        │
    │ - Combined Tables     │                    │ /optimize (POST)     │
    └──────────────────────┘                    └──────────────────────┘
        │                                                   │
        └───────────────────────┬──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Data Storage Layer  │
                    │                      │
                    │ - DuckDB (local)     │
                    │ - Parquet Files      │
                    │ - Model Checkpoints  │
                    │ - Training Logs      │
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  External Data       │
                    │                      │
                    │ - yfinance (OHLCV)   │
                    │ - News APIs          │
                    │ - Twitter/Reddit     │
                    │ - Financial Feeds    │
                    └──────────────────────┘
```

---

## Core Components

### 1. **Orchestrator** (`src/pipeline/orchestrator.py`)
Central coordinator for all pipeline operations.

**Responsibilities:**
- Load and validate configuration
- Manage conditional feature generation
- Handle cached feature loading
- Trigger dashboard-integrated forecasting
- Ensure idempotent runs via task markers

**Key Methods:**
- `prepare_features(ticker)` - Build/cache features
- `train_model()` - Execute training pipeline
- `forecast_equity(ticker)` - Generate predictions
- `optimize_hyperparameters()` - Run Optuna trials
- `ensemble_predictions()` - Combine model outputs

### 2. **DAG Runner** (`src/dag/dag_runner.py`)
Graph-based pipeline executor with dependency management.

**Responsibilities:**
- Topological traversal of pipeline stages
- Dependency checking before stage execution
- Retry logic with exponential backoff
- State tracking (RUNNING, SUCCESS, FAILED)
- Monitoring hook integration

**Key Methods:**
- `run()` - Execute DAG in topological order
- `run_stage(stage_name)` - Run single stage with retry

### 3. **State Manager** (`src/dag/state_manager.py`)
Persistent state tracking for pipeline stages.

**Responsibilities:**
- Track stage execution status
- Write/read state markers to filesystem
- Detect stale states
- Support recovery from failures

**Files Tracked:**
- `datalake/{stage}/.{stage}_RUNNING`
- `datalake/{stage}/.{stage}_SUCCESS`
- `datalake/{stage}/.{stage}_FAILED`

### 4. **Data Validation** (`src/validation/`)

**Components:**
- **expectations.py** - Data quality rules (Great Expectations integration)
- **validator.py** - Pipeline validation at each stage
- **sanitizer.py** - Input cleaning and normalization

**Validation Stages:**
- Column existence and types
- Value ranges (price > 0, volume >= 0)
- NaN/null percentage limits
- Duplicate detection
- Datetime continuity

### 5. **Configuration System** (`src/config/`)

**Files:**
- `config.yaml` - Master configuration
- `config_loader.py` - YAML parsing with type validation
- `runtime_config.py` - Runtime parameter overrides
- `active_equity.py` - Current trading symbol tracking

**Supported Parameters:**
- Data paths
- Model hyperparameters
- Training settings (batch size, epochs, learning rate)
- Validation parameters
- API configuration

---

## Data Flow Pipeline

### High-Level Flow

```
Raw Market Data → Preprocessing → Feature Engineering → Model Training
                                                              ↓
                                                    Hyperparameter Optimization
                                                              ↓
                                                    Model Evaluation & Selection
                                                              ↓
                                                    Ensemble Meta-Learning
                                                              ↓
                                                    Inference & Prediction
                                                              ↓
                                                    API/Dashboard Serving
```

### Detailed Data Transformation

```
Stage A (Ingestion):
  Input:  Stock symbols, date range
  Output: Raw OHLCV dataframes
  Cache:  datalake/cache/data/{symbol}_raw.parquet

Stage B (Preprocessing):
  Input:  Raw OHLCV
  Output: Cleaned OHLCV (outliers removed, NaNs handled)
  Cache:  datalake/cache/data/{symbol}_preprocessed.parquet

Stage C (Feature Generation):
  Input:  Preprocessed OHLCV + Global signals + Sentiment
  Output: Feature matrices (50-100 features per symbol)
  Features:
    - Technical: MA, RSI, MACD, Bollinger Bands, ATR, etc.
    - Momentum: ROC, Stochastic Oscillator
    - Volatility: Standard Deviation, Historical Volatility
    - Sentiment: News sentiment scores, Twitter sentiment
    - Global: Cross-equity correlations, market momentum
  Cache:  datalake/cache/features/{symbol}_features.joblib

Stage D (Optimization):
  Input:  Features, historical prices
  Output: Optimal hyperparameters for each model
  Method: Optuna Bayesian optimization
  Cache:  datalake/experiments/optuna/study.db

Stage E (Model Training):
  Input:  Optimized features, price targets
  Output: Trained model weights
  Models: LSTM, TCN, XGBoost, LightGBM, Ridge, ElasticNet
  Cache:  datalake/models/trained/{model}_{symbol}.pth

Stage F (Inference):
  Input:  Latest features
  Output: Individual model predictions
  Cache:  datalake/predictions/latest/{model}_{symbol}.npy

Stage G (Walk-Forward Validation):
  Input:  Training + validation feature windows
  Output: Out-of-fold predictions for meta-learning
  Method: Time-series cross-validation windows
  Cache:  datalake/ensemble/oof_{symbol}.joblib

Stage H (Ensemble):
  Input:  Individual model predictions + OOF predictions
  Output: Ensemble meta-learner
  Method: Stacking (meta-learner trained on OOF)
  Cache:  datalake/ensemble/meta_model.joblib
```

---

## Pipeline Stages (A-H)

### Stage A: Data Ingestion Pipeline
**File:** `src/pipeline/A_ingestion_pipeline.py`

**Purpose:** Download and cache market data

**Process:**
1. Download OHLCV data from yfinance
2. Validate column existence and types
3. Check for NaN/duplicate rows
4. Store in cacheable format (Parquet)
5. Log data statistics

**Output Schema:**
```
Date (datetime64)  | Open (float64)  | High (float64)  | Low (float64)  | Close (float64)  | Volume (int64)
```

**Caching:**
- Location: `datalake/cache/data/{symbol}_raw.parquet`
- Reuse: Same date range detection

---

### Stage B: Preprocessing Pipeline
**File:** `src/pipeline/B_preprocessing_pipeline.py`

**Purpose:** Clean and normalize data

**Process:**
1. Convert to numeric types (handle errors)
2. Remove/interpolate NaN values
3. Detect and remove outliers (IQR method)
4. Resample to daily frequency
5. Sort by date
6. Validate price ranges

**Outlier Detection:**
```
Price changed > 10% in single day? Investigate.
Volume > 3x median? Suspicious.
Price <= 0? Invalid.
```

**Output Schema:** Same as Stage A (cleaned)

---

### Stage C: Feature Generation Pipeline
**File:** `src/pipeline/C_feature_gen_pipeline.py`

**Purpose:** Engineer features from raw OHLCV

**Features Generated:**

**Technical Indicators** (20+ features):
- Moving Averages: SMA(10,20,50), EMA(10,20)
- Momentum: RSI, MACD, ROC
- Volatility: Standard Deviation, ATR
- Oscillators: Stochastic, CCI
- Bands: Bollinger Bands

**Sentiment Features** (5+ features):
- News sentiment score (0-1)
- Social media sentiment
- Prominence score
- Time decay (recent news weighted more)

**Global Signals** (10+ features):
- Market-wide momentum
- Correlation with SPY/QQQ
- Sector performance
- Cross-equity patterns

**Output Size:**
- ~80-100 features per symbol
- Rolling windows of 30, 60, or custom lookback

---

### Stage D: Hyperparameter Optimization Pipeline
**File:** `src/pipeline/D_optimization_pipeline.py`

**Purpose:** Find optimal model parameters

**Method:** Optuna (Bayesian Optimization)

**Search Space:**
```
LSTM:
  - hidden_size: [32, 128]
  - num_layers: [1, 3]
  - dropout: [0.1, 0.5]
  - learning_rate: [1e-4, 1e-2]

XGBoost:
  - max_depth: [3, 10]
  - learning_rate: [0.01, 0.3]
  - n_estimators: [100, 500]

LightGBM:
  - num_leaves: [20, 150]
  - learning_rate: [0.01, 0.3]
  - min_child_samples: [5, 30]
```

**Evaluation:** Minimize RMSE on validation set

**Output:** Best parameters stored in SQLite database

---

### Stage E: Model Training Pipeline
**File:** `src/pipeline/E_modeltrainer_pipeline.py`

**Purpose:** Train model ensemble on optimized parameters

**Models Trained:**
1. LSTM (RNN) - Sequence learning
2. TCN (Temporal Convolutional) - Parallel convolutional learning
3. XGBoost - Gradient boosting
4. LightGBM - Fast gradient boosting
5. Ridge Regression - L2 regularization
6. ElasticNet - L1+L2 regularization

**Training Configuration:**
- Train/val/test split: 60%/20%/20%
- Batch size: 32 (configurable)
- Epochs: 100-500 (early stopping)
- Loss: MSE or custom weighted loss
- Optimizer: Adam (for neural nets)

**Output:**
- Model weights: `datalake/models/trained/{model}_{symbol}.pth`
- Training logs: tensorboard events

---

### Stage F: Inference Pipeline
**File:** `src/pipeline/F_inference_pipeline.py`

**Purpose:** Generate predictions using trained models

**Process:**
1. Load latest features for symbol
2. Load trained model weights
3. Forward pass through model
4. Generate point predictions
5. (Optional) Confidence intervals with Monte Carlo

**Output:**
- Predictions: `datalake/predictions/latest/{model}_{symbol}.npy`
- Format: 1D array of next-step or multi-step forecasts
- Timestamp: Prediction generation time

---

### Stage G: Walk-Forward Validation Pipeline
**File:** `src/pipeline/G_wfv_pipeline.py`

**Purpose:** Generate out-of-fold predictions for meta-learning

**Method:** Time-series walk-forward cross-validation

```
┌─────────────────────────────────────────────────────────┐
│ Full Historical Data (5 years)                          │
├─────────────────────────────────────────────────────────┤
│ Train        │ Val │ Test   (Fold 1)                    │
│              │     │                                    │
│        Train        │ Val │ Test   (Fold 2)             │
│                     │     │                             │
│               Train        │ Val │ Test   (Fold 3)      │
│                            │     │                      │
│                      Train       │ Val │ Test (Fold 4)  │
└─────────────────────────────────────────────────────────┘
                         ↓
                OOF matrix (full dataset predictions)
                    Used for meta-learning
```

**Output:**
- OOF predictions: `datalake/ensemble/oof_{symbol}.joblib`
- Shape: (num_samples, num_models)
- Each row: predictions from all base models at time t

---

### Stage H: Ensemble Pipeline
**File:** `src/pipeline/H_ensemble_pipeline.py`

**Purpose:** Learn optimal model combination using meta-learning

**Ensemble Methods:**

**1. Simple Ensembler:**
- Mean: Average of all predictions
- Median: Median of all predictions
- Weighted: Custom weights per model

**2. Meta-Learner (Stacking):**
- Input: OOF predictions from all base models
- Meta-model: LightGBM or linear layer
- Training: Learn to optimally weight base models
- Output: Single ensemble prediction

**Meta-Model Selection:**
```
if num_samples > 1000:
    use LightGBM (captures non-linearity)
else:
    use linear regression (robust to small data)
```

**Output:**
- Meta-model weights: `datalake/ensemble/meta_model.joblib`
- Ensemble predictions: `datalake/predictions/latest/ensemble_{symbol}.npy`

---

## Models Architecture

### 1. LSTM Model
**File:** `src/models/lstm_model.py`

**Architecture:**
```
Input Features (batch_size, seq_len, input_size)
    ↓
LSTM Cell 1 (hidden_size=64)
    ↓
Dropout (p=0.3)
    ↓
LSTM Cell 2 (hidden_size=64)
    ↓
Dropout (p=0.3)
    ↓
Dense Layer (input_size → 1)
    ↓
Output Prediction (scalar)
```

**Hyperparameters:**
- `input_size`: 80-100 (number of features)
- `hidden_size`: 32-128 (optimized)
- `num_layers`: 1-3 (optimized)
- `dropout`: 0.1-0.5 (optimized)
- `output_size`: 1 (single-step forecast)

**Training:**
- Loss: MSE
- Optimizer: Adam
- Learning rate: 1e-4 to 1e-2 (optimized)
- Epochs: 100-500 with early stopping

**Use Case:** Time-series sequence modeling, captures temporal dependencies

---

### 2. TCN Model (Temporal Convolutional Network)
**File:** `src/models/tcn_model.py`

**Architecture:**
```
Input Features (batch_size, seq_len, input_size)
    ↓
1D Convolution (kernel_size=3, dilation=1)
    ↓
Residual Blocks (multiple)
    ↓
1D Convolution (kernel_size=3, dilation=2^i)
    ↓
Dropout + Activation
    ↓
Global Average Pooling
    ↓
Dense Output Layer
    ↓
Output Prediction
```

**Advantages over LSTM:**
- Parallelizable (no sequential dependencies)
- Captures long-range patterns with dilated convolutions
- Generally faster training

**Use Case:** When computational efficiency required, parallel processing available

---

### 3. XGBoost Model
**File:** `src/models/lightGBM_model.py`

**Architecture:**
```
Tree Ensemble (100-500 trees)
Each tree:
  - Max depth: 3-10
  - Learning rate: 0.01-0.3
  - Handles non-linear feature interactions
  - gradient-based split selection
```

**Feature Importance:** Models feature contributions directly

**Use Case:** Non-linear patterns, feature engineering not required, robust

---

### 4. LightGBM Model
**File:** `src/models/lightGBM_model.py`

**Architecture:**
```
Leaf-wise tree growth (vs XGBoost's level-wise)
Histogram-based learning (faster)
Multiple iterations with residuals
GOSS: Gradient-based One-Side Sampling
```

**Advantages:**
- 10-20x faster training than XGBoost
- Lower memory usage
- Similar accuracy
- Works well with large datasets

**Use Case:** Production models where speed is critical

---

### 5. Ridge Regression
**File:** `src/models/ridge_regg_model.py`

**Architecture:**
```
Linear Model with L2 Regularization
Loss = MSE + λ * sum(weights²)
Closed-form solution or iterative optimization
```

**Regularization Parameter:** α ∈ [0.001, 100]

**Use Case:** Baseline, interpretable model, preventing overfitting

---

### 6. ElasticNet
**File:** `src/models/elastic_net_model.py`

**Architecture:**
```
Linear Model with L1+L2 Regularization
Loss = MSE + λ₁ * sum(|weights|) + λ₂ * sum(weights²)
Combines Lasso (L1) and Ridge (L2) benefits
```

**Use Case:** Feature selection + regularization balance

---

### Base Model Class
**File:** `src/models/base_model.py`

**Common Interface:**
```python
class BaseModel:
    def train(X_train, y_train, X_val, y_val) → metrics
    def predict(X) → predictions
    def save(path) → None
    def load(path) → None
    def get_hyperparams() → dict
    def set_hyperparams(dict) → None
```

**Standardizes:**
- Training loop
- Validation logic
- Prediction interface
- Serialization

---

## Ensemble Strategy

### Why Ensemble?

**Problem:** Single model has blind spots
- LSTM excels at sequences but poor at feature interactions
- XGBoost captures interactions but ignores time dependencies
- Linear models are interpretable but miss non-linearity

**Solution:** Combine predictions to leverage each model's strength

### Ensemble Architecture

```
╔═══════════════════════════════════════════════════╗
║         BASE MODELS (Stage E Output)              ║
║  LSTM │ TCN │ XGBoost │ LGBM │ Ridge │ ElasticNet ║
╚═══════════════════════════════════════════════════╝
           ↓         ↓         ↓        ↓       ↓
          (1)       (2)       (3)      (4)     (5) Individual Predictions


╔═══════════════════════════════════════════════════╗
║   OUT-OF-FOLD PREDICTIONS (Stage G Output)       ║
║   Predictions on training/validation data         ║
║   from each base model                            ║
╠═══════════────────(n_samples × 6)═════════════════╣
║ [pred₁_LSTM] [pred₁_TCN] [pred₁_XGB] ...        ║
║ [pred₂_LSTM] [pred₂_TCN] [pred₂_XGB] ...        ║
║ ...                                               ║
║ [predₙ_LSTM] [predₙ_TCN] [predₙ_XGB] ...        ║
╚═══════════════════════════════════════════════════╝
                      ↓
╔═══════════════════════════════════════════════════╗
║        META-LEARNER TRAINING (Stage H)            ║
║                                                   ║
║  Input:  OOF predictions (6 features)            ║
║  Target: Actual price changes                    ║
║  Model:  LightGBM (learns optimal combining)     ║
║                                                   ║
║  Learns:                                          ║
║  - Which models are reliable in different        ║
║    market conditions                             ║
║  - How much to weight each model's prediction   ║
║  - Non-linear combinations of models            ║
╚═══════════════════════════════════════════════════╝
                      ↓
╔═══════════════════════════════════════════════════╗
║           FINAL ENSEMBLE PREDICTION               ║
║                                                   ║
║  ensemble_pred = meta_model([p_LSTM, p_TCN,     ║
║                               p_XGB, p_LGBM,     ║
║                               p_Ridge, p_ElNet]) ║
║                                                   ║
║  Output: Single best prediction combining all    ║
║          models' strengths                       ║
╚═══════════════════════════════════════════════════╝
```

### Ensemble Methods Available

**1. Simple Average:**
```
ensemble = mean([LSTM, TCN, XGB, LGBM, Ridge, ElNet])
```

**2. Weighted Average:**
```
ensemble = sum([w_i * pred_i for i in models])
weights = [0.2, 0.15, 0.25, 0.25, 0.08, 0.07]
```

**3. Stacking (Meta-Learner):**
```
meta_model = train_on_oof_predictions()
ensemble = meta_model([pred_LSTM, pred_TCN, ...])
```

**4. Voting (Classification):**
```
# For classification (up/down/sideways)
ensemble = mode([pred_LSTM, pred_TCN, ...])
```

### Flow in Code

**Training (Stage H):**
```python
# Generate OOF predictions (Stage G)
oof_preds = generate_oof_predictions()  # (n_samples, 6)

# Train meta-model on OOF
meta_model = train_meta_model(oof_preds, y_true)

# Save for inference
save_model(meta_model)
```

**Inference:**
```python
# Get latest predictions from all base models
preds = {
    'lstm': lstm_model.predict(X),
    'tcn': tcn_model.predict(X),
    ...
}

# Stack predictions
stacked_pred = np.column_stack([preds['lstm'], preds['tcn'], ...])

# Meta-model inference
ensemble_pred = meta_model.predict(stacked_pred)
```

---

## API Layer

### Endpoints

**File:** `src/api/main_api.py`, `src/api/forecasting_api.py`, `src/api/training_api.py`

**Architecture:**
```
FastAPI App
├── Forecasting Router
│   ├── POST /forecast
│   ├── GET /forecast/{symbol}
│   └── POST /backtest
├── Sentiment Router
│   ├── POST /sentiment
│   └── GET /sentiment/{symbol}
├── Training Router
│   ├── POST /train
│   ├── POST /optimize
│   └── GET /train/status
└── Middleware
    ├── CORS
    ├── Logging
    └── Error handling
```

### Forecast Endpoint

**POST /forecast**

**Request:**
```json
{
  "equity": "AAPL",
  "horizon": 10,
  "features": ["technical", "sentiment", "global"],
  "ensemble_method": "stacking"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2026-03-18T10:30:00",
  "forecast": [150.2, 151.3, 152.1, ...],
  "confidence_interval": [[149.5, 150.9], [150.6, 152.0], ...],
  "base_model_predictions": {
    "lstm": [150.1, 151.2, ...],
    "xgboost": [150.3, 151.5, ...],
    ...
  },
  "ensemble_weights": {
    "lstm": 0.20,
    "tcn": 0.15,
    ...
  }
}
```

### Sentiment Endpoint

**POST /sentiment**

**Request:**
```json
{
  "symbols": ["AAPL", "MSFT"],
  "lookback_days": 7,
  "data_sources": ["news", "twitter"]
}
```

**Response:**
```json
{
  "timestamp": "2026-03-18T10:30:00",
  "sentiment": {
    "AAPL": {
      "score": 0.65,
      "magnitude": 0.5,
      "sources": {
        "news": 0.70,
        "twitter": 0.60
      }
    },
    "MSFT": {...}
  }
}
```

### Training Endpoint

**POST /train**

**Request:**
```json
{
  "symbols": ["AAPL"],
  "lookback_days": 252,
  "models": ["lstm", "xgboost", "lgbm"],
  "optimize": true
}
```

**Response:**
```json
{
  "job_id": "train_20260318_103000",
  "status": "TRAINING",
  "models_training": ["lstm", "xgboost"],
  "progress": 0.35,
  "eta_seconds": 1800
}
```

---

## Dashboard Integration

### Technology Stack
- **Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **State Management:** Streamlit session state

### Components

**File:** `src/dashboard/app.py`

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  EQUITY FORECASTING DASHBOARD                                  │
├──────────────┬────────────────────────────────────────────────┤
│              │                                                │
│   SIDEBAR    │              MAIN CONTENT                     │
│              │                                                │
│ • Select     │  ┌────────────────────────────────────────┐  │
│   Equity     │  │     Forecast Panel                     │  │
│ • Horizon    │  │                                        │  │
│ (1-30 days)  │  │  - Line chart (forecast + history)    │  │
│              │  │  - Confidence bounds                  │  │
│ • Data View  │  │  - Model contributions                │  │
│   - Forecast │  │                                        │  │
│   -Sentiment │  └────────────────────────────────────────┘  │
│   - Both     │                                                │
│              │  ┌────────────────────────────────────────┐  │
│ • Refresh    │  │     Sentiment Panel                    │  │
│   (Real-time)│  │                                        │  │
│              │  │  - Sentiment score gauge               │  │
│              │  │  - News/Twitter breakdown              │  │
│              │  │  - Recent headlines                    │  │
│              │  └────────────────────────────────────────┘  │
│              │                                                │
│              │  ┌────────────────────────────────────────┐  │
│              │  │     History Manager                    │  │
│              │  │                                        │  │
│              │  │  - Volume/Price history                │  │
│              │  │  - Trading statistics                  │  │
│              │  │  - Download CSV                        │  │
│              │  └────────────────────────────────────────┘  │
└──────────────┴────────────────────────────────────────────────┘
```

### Key Components

**1. Forecast Panel** (`src/dashboard/forecast_panel.py`)
- Renders forecast line chart
- Shows confidence intervals
- Displays individual model predictions
- Updates via API calls to orchestrator

**2. Sentiment Panel** (`src/dashboard/sentiment_panel.py`)
- Displays sentiment gauge (0-1 scale)
- Shows source breakdown (news vs social)
- Lists recent headlines
- Updates forecast with sentiment data

**3. Combined Table** (`src/dashboard/combined_tabel.py`)
- Historical price + indicators
- Volume, returns, volatility
- Downloadable as CSV

**4. History Manager** (`src/dashboard/history_manager.py`)
- Manages historical data for symbols
- Caches locally
- Handles CSV export

### Workflow

```
User selects equity + horizon
          ↓
render_sidebar() → Input validation
          ↓
set_active_equity() → Validate ticker
          ↓
prepare_features() → Load/generate features
          ↓
forecast_panel.render_forecast()
  ├── Call API → /forecast
  ├── Get predictions + confidence
  └── Render interactive chart
          ↓
(Optional) sentiment_panel.render_sentiment()
  ├── Call API → /sentiment
  └── Display sentiment breakdown
          ↓
Display to user + allow refresh
```

---

## Configuration System

### Configuration Files

**`src/config/config.yaml`** - Master configuration
```yaml
data:
  raw_data_dir: "datalake/data/raw/"
  processed_data_dir: "datalake/data/processed/"
  features_data_dir: "datalake/data/features/"

model:
  checkpoints_dir: "datalake/models/checkpoints/"
  trained_model_dir: "datalake/models/trained/"
  trained_model_file_pattern: "lstm_global_{num_equities}eqt_{MMDD}.pth"

model_params:
  input_size: 10
  hidden_size: 64
  num_layers: 2
  output_size: 1
  dropout: 0.3
  learning_rate: 0.001
  batch_size: 32

training:
  epochs: 100
  early_stopping_patience: 15
  train_val_test_split: [0.6, 0.2, 0.2]
```

**`src/config/config_loader.py`** - Type-safe loading
```python
def load_typed_config(path: str) -> FullConfig:
    """Load and validate config with Pydantic models"""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FullConfig(**raw)
```

**`src/config/runtime_config.py`** - CLI overrides
```python
# Usage: main.py --learning_rate 0.002 --hidden_size 128
```

**`src/config/active_equity.py`** - Current symbol tracking
```python
set_active_equity("AAPL")  # Used by dashboard
get_active_equity()  # Returns current symbol
```

### Configuration Flow

```
main.py [--config path] [--param value]
        ↓
config_loader.load_typed_config()
        ↓
runtime_config.apply_overrides()
        ↓
FullConfig object (validated)
        ↓
Pass to Orchestrator/DAG/API
```

---

## Validation & Monitoring

### Data Validation Pipeline

**File:** `src/validation/validator.py`

**Stages:**

**1. Input Sanitization** (Before ingestion)
```python
# Convert column names to lowercase
# Check required columns exist: OHLCV
# Remove non-numeric values
# Handle missing data
```

**2. Range Validation** (After preprocessing)
```python
# Close price > 0
# Volume >= 0
# High >= Low >= Close >= Open
# No NaN/Inf values
# No duplicate dates
```

**3. Feature Validation** (After feature generation)
```python
# Check feature shape matches expected
# No NaN values in features
# Feature values in reasonable ranges
# Sufficient variance in features
```

### Monitoring & Logging

**File:** `src/monitoring/monitor.py` and `src/utils/logger.py`

**Logging Levels:**
```
DEBUG   - Detailed execution flow
INFO    - Significant events (stage completion)
WARNING - Recoverable issues (missing data)
ERROR   - Unrecoverable problems
CRITICAL - System failures
```

**Logged Information:**
- Pipeline stage start/completion
- Model training metrics (loss, accuracy)
- Predictions and confidence
- Error traces with context
- Performance metrics (training time, inference latency)

**Log Outputs:**
- Console (real-time)
- File: `datalake/logs/train.log`
- File: `datalake/logs/inference.log`
- (Future) CloudWatch/Datadog integration

---

## File Structure

### Detailed Directory Map

```
equity_forecasting/
│
├── main.py                          # CLI entrypoint
├── requirements.txt                 # Python dependencies
├── pytest.ini                        # Test configuration
│
├── src/
│   ├── __init__.py
│   │
│   ├── adapter/                     # External service adapters
│   │   ├── adapter.py               # Global signal adapter
│   │   └── __pycache__/
│   │
│   ├── api/                         # REST API layer
│   │   ├── main_api.py              # FastAPI app + middleware
│   │   ├── forecasting_api.py       # /forecast endpoints
│   │   ├── sentiment_api.py         # /sentiment endpoints
│   │   ├── training_api.py          # /train endpoints
│   │   ├── main.py                  # API startup
│   │   └── __pycache__/
│   │
│   ├── config/                      # Configuration management
│   │   ├── config.yaml              # Master config
│   │   ├── config_loader.py         # YAML parsing + validation
│   │   ├── runtime_config.py        # CLI parameter overrides
│   │   ├── active_equity.py         # Current symbol tracking
│   │   ├── api_models.py            # Pydantic models (deprecated?)
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   ├── dag/                         # DAG-based orchestration
│   │   ├── dag_config.py            # DAG configuration
│   │   ├── dag_errors.py            # Custom exceptions
│   │   ├── dag_graph.py             # Graph representation
│   │   ├── dag_node.py              # Node abstraction
│   │   ├── dag_runner.py            # Graph executor
│   │   ├── dag_stages.py            # Stage implementations
│   │   ├── state_manager.py         # Persistent state tracking
│   │   ├── testdag.py               # DAG testing
│   │   ├── equity_sets/             # Equity group definitions
│   │   ├── runners/                 # Alternative runners
│   │   └── __pycache__/
│   │
│   ├── dashboard/                   # Streamlit UI
│   │   ├── app.py                   # Main dashboard entry
│   │   ├── ui_components.py         # Reusable UI widgets
│   │   ├── forecast_panel.py        # Forecast visualization
│   │   ├── sentiment_panel.py       # Sentiment display
│   │   ├── combined_tabel.py        # Historic data table
│   │   ├── history_manager.py       # Historical data cache
│   │   ├── utils.py                 # Dashboard utilities
│   │   └── __pycache__/
│   │
│   ├── data_yfin/                   # Data loading utilities
│   │   ├── load_api_data.py         # yfinance API wrapper
│   │   ├── load_csv.py              # CSV data loading
│   │   └── __pycache__/
│   │
│   ├── ensemble/                    # Ensemble meta-learning
│   │   ├── __init__.py
│   │   ├── simple_ensembler.py      # Mean/median/weighted
│   │   ├── generate_oof.py          # OOF prediction generation
│   │   ├── meta_features.py         # Meta-feature engineering
│   │   ├── train_meta_features.py   # Meta-model training
│   │   ├── evaluate_meta_model.py   # Meta-model evaluation
│   │   └── __pycache__/
│   │
│   ├── features/                    # Feature engineering
│   │   ├── market_sentiment/        # Sentiment feature extraction
│   │   │   └── sentiment_analyzer.py
│   │   ├── technical/               # Technical indicator conversion
│   │   │   └── indicators.py
│   │   └── __init__.py
│   │
│   ├── global_signal/               # Global market patterns
│   │   ├── __init__.py
│   │   ├── commands.py              # Training/inference
│   │   ├── global_signal.npy        # Cached signal file
│   │   └── promote_global_signal.ipynb
│   │
│   ├── models/                      # Base models architecture
│   │   ├── base_model.py            # Abstract base class
│   │   ├── lstm_model.py            # LSTM implementation
│   │   ├── tcn_model.py             # Temporal CNN
│   │   ├── xgboost_model.py         # XGBoost wrapper
│   │   ├── lightGBM_model.py        # LightGBM wrapper
│   │   ├── ridge_regg_model.py      # Ridge regression
│   │   ├── elastic_net_model.py     # ElasticNet
│   │   └── __pycache__/
│   │
│   ├── monitoring/                  # Metrics & observability
│   │   ├── monitor.py               # Training monitor
│   │   └── __pycache__/
│   │
│   ├── optimizers/                  # Optimization algorithms
│   │   └── (custom optimizers if needed)
│   │
│   ├── pipeline/                    # Pipeline orchestration
│   │   ├── A_ingestion_pipeline.py   # Data download & ingestion
│   │   ├── B_preprocessing_pipeline.py # Data cleaning
│   │   ├── C_feature_gen_pipeline.py  # Feature engineering
│   │   ├── D_optimization_pipeline.py # Hyperparameter tuning
│   │   ├── E_modeltrainer_pipeline.py # Model training
│   │   ├── F_inference_pipeline.py    # Prediction generation
│   │   ├── G_wfv_pipeline.py          # Walk-forward validation
│   │   ├── H_ensemble_pipeline.py     # Ensemble meta-learning
│   │   ├── orchestrator.py            # Master orchestrator
│   │   ├── pipeline_wrapper.py        # Wrapper utilities
│   │   ├── state_manager.py           # State tracking
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   ├── training/                    # Training utilities
│   │   └── (training loops, loss functions)
│   │
│   ├── utils/                       # Utility functions
│   │   ├── cache_manager.py         # Cache handling
│   │   ├── config.py                # Config utilities
│   │   ├── config_loader.py         # Config loading
│   │   ├── logger.py                # Logging setup
│   │   ├── model_utils.py           # Model utilities
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   └── validation/                  # Data validation
│       ├── expectations.py          # Great Expectations rules
│       ├── validator.py             # Main validator
│       ├── sanitizer.py             # Input cleaning
│       ├── __init__.py
│       └── __pycache__/
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_*.py                    # Unit tests
│   ├── conftest.py                  # Pytest fixtures
│   ├── data/                        # Test data
│   ├── mocks/                       # Mock objects
│   └── __pycache__/
│
├── datalake/                        # Data storage (auto-created)
│   ├── data/
│   │   ├── raw/                     # Raw OHLCV data
│   │   ├── processed/               # Cleaned data
│   │   │   └── cache/
│   │   └── features/
│   ├── cache/                       # Feature cache
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   ├── ensemble/
│   │   └── forecasting/
│   ├── models/
│   │   ├── trained/                 # Trained model weights
│   │   ├── checkpoints/             # Training checkpoints
│   │   └── global/
│   ├── predictions/
│   │   ├── test/
│   │   ├── validation/
│   │   ├── latest/                  # Latest predictions
│   │   └── production/
│   ├── experiments/
│   │   ├── optuna/
│   │   │   └── study.db             # Optuna trial database
│   │   └── wandb/
│   ├── ensemble/
│   │   ├── oof_*.joblib             # Out-of-fold predictions
│   │   └── meta_model.joblib        # Trained meta-learner
│   ├── runs/
│   │   ├── error_log.jsonl
│   │   └── RUN_20251224_153655/
│   ├── metadata/
│   │   ├── schema/
│   │   └── stats/
│   ├── logs/
│   │   ├── train.log
│   │   └── inference.log
│   └── evaluation/
│       ├── metrics/
│       └── reports/
│
├── notebooks/                       # Jupyter notebooks (development)
│   ├── blueprint.ipynb
│   ├── download_stocks.ipynb
│   ├── project_blueprint.py
│   ├── project_overview.ipynb
│   ├── file_extractor.ipynb
│   ├── just_test.ipynb
│   ├── yfinance_rate_test.ipynb
│   └── yfinance_stress_test.ipynb
│
├── Documentation files:
│   ├── ARCHITECTURE.md               # This file
│   ├── README_DEPLOYMENT.md          # AWS deployment guide
│   ├── FILE_MODIFICATION_GUIDE.md    # Implementation checklist
│   ├── IMPLEMENTATION_ROADMAP.md     # Development roadmap
│   └── TRACKING_CHECKLIST.md         # Daily progress tracker
│
└── Configuration files:
    ├── .env                          # Environment variables
    ├── pyproject.toml                # Project metadata
    └── pytest.ini                    # Test configuration
```

---

## Dependencies & Versioning

### Core Dependencies

**Data Processing:**
- `numpy==2.4.2` - Numerical computing
- `pandas==2.2.3` - DataFrames and data manipulation
- `scikit-learn==1.6.1` - ML algorithms and preprocessing

**Deep Learning:**
- `pytorch==2.6.0` - Deep learning framework
- `pytorch-lightning==2.4.0` - High-level PyTorch API

**Gradient Boosting:**
- `xgboost==2.1.1` - XGBoost implementation
- `lightgbm==4.1.1` - LightGBM implementation

**Financial Data:**
- `yfinance==0.2.33` - Yahoo Finance data download

**API & Web:**
- `fastapi==0.115.12` - REST API framework
- `uvicorn==0.31.0` - ASGI server
- `pydantic==2.11.1` - Data validation

**Optimization:**
- `optuna==4.1.0` - Hyperparameter optimization

**NLP & Sentiment:**
- `nltk==3.8.1` - Natural language toolkit
- `textblob==0.17.1` - Simple NLP
- `transformers==4.48.2` - Pre-trained transformers
- `feedparser==6.0.11` - RSS feed parsing

**Visualization:**
- `matplotlib==3.10.1` - Static plots
- `plotly==5.24.1` - Interactive plots
- `seaborn==0.13.2` - Statistical plots

**Storage:**
- `sqlalchemy==2.0.35` - SQL toolkit
- `pyarrow==17.0.0` - Apache Arrow integration

**Web Scraping:**
- `beautifulsoup4==4.13.4` - HTML parsing
- `requests==2.32.3` - HTTP library
- `httpx==0.28.1` - Async HTTP client

**Logging & Monitoring:**
- `python-json-logger==3.2.1` - JSON logging

**Utilities:**
- `python-dotenv==1.0.1` - Environment variables
- `tqdm==4.67.1` - Progress bars
- `schedule==1.2.2` - Task scheduling

### Development Dependencies

```
pytest>=7.0
pytest-cov>=4.0
black>=22.0
mypy>=0.990
flake8>=4.0
```

---

## Design Patterns

### 1. **Pipeline Pattern**
Stages A-H form a data processing pipeline where each stage:
- Consumes output of previous stage
- Performs transformation
- Produces output for next stage
- Can be cached independently

**Benefit:** Modularity, reusability, easy testing

### 2. **DAG Pattern**
Pipeline execution managed as directed acyclic graph:
- Nodes = pipeline stages
- Edges = dependencies
- Topological sort = execution order
- Allows parallel execution of independent stages

**Benefit:** Flexibility, scalability, restart capability

### 3. **State Manager Pattern**
Track pipeline progress with filesystem markers:
- `{stage}_RUNNING` - Currently executing
- `{stage}_SUCCESS` - Completed
- `{stage}_FAILED` - Error occurred

**Benefit:** Idempotent runs, crash recovery, monitoring

### 4. **Strategy Pattern**
Multiple ensemble strategies encapsulated:
- `SimpleEnsembler` - Mean/median/weighted
- `StackingEnsembler` - Meta-learner
- `VotingEnsembler` - Majority vote

**Benefit:** Easy to add new ensemble methods, testable

### 5. **Factory Pattern**
Model creation as factory methods:
```python
def create_model(model_type: str, **kwargs) -> BaseModel:
    models = {
        'lstm': LSTMModel,
        'xgboost': XGBoostModel,
        'lgbm': LightGBMModel,
    }
    return models[model_type](**kwargs)
```

**Benefit:** Decoupled instantiation, easier configuration

### 6. **Adapter Pattern**
External integrations wrapped in adapters:
- `DataAdapter` - yfinance API
- `SentimentAdapter` - NLP models
- `APIAdapter` - Chart plotting

**Benefit:** Loose coupling, easy to replace implementations

### 7. **Observer Pattern** (Future)
Monitoring hooks for pipeline events:
```python
orchestrator.on('stage_complete', log_metrics)
orchestrator.on('error', send_alert)
```

---

## Current Status & Known Issues

### Status Summary
- **Overall:** 30% production-ready
- **Data Pipeline:** 85% complete (validation adding ~15%)
- **Models:** 75% complete (need type checking, error handling)
- **API:** 60% complete (needs Pydantic validation)
- **Dashboard:** 50% complete (basic UI, needs sentinel features)
- **Monitoring:** 20% complete (basic logging only)

### Critical Issues (Must Fix Before AWS)

**1. Data Validation Missing** ⚠️ HIGH
- Problem: Pipeline accepts invalid data without checks
- Impact: Silent corruption → biased models
- Solution: Implement Great Expectations framework (3 hrs)
- Files: Create `src/validation/expectations.py`, `validator.py`
- Status: IN PROGRESS

**2. Type Safety** ⚠️ HIGH
- Problem: Runtime type errors (NaN, Inf not caught)
- Impact: Model training crashes
- Solution: Add type checking in adapter.py, lstm_model.py (2 hrs)
- Files: `src/adapter/adapter.py`, `src/models/lstm_model.py`
- Status: NOT STARTED

**3. API Input Validation** ⚠️ MEDIUM
- Problem: REST endpoints don't validate requests
- Impact: Malformed requests cause crashes
- Solution: Implement Pydantic models (1 hr)
- Files: `src/api/models.py` (create new)
- Status: NOT STARTED

**4. Error Handling** ⚠️ MEDIUM
- Problem: Exceptions not caught, no graceful degradation
- Impact: Pipeline crashes on edge cases
- Solution: Add try-catch blocks, logging (2 hrs)
- Files: All pipeline stages
- Status: PARTIAL

### High Priority (Next 2 weeks)

- **MLflow Integration** - Model versioning and tracking
- **Structured Logging** - JSON logging for observability
- **Database State** - Replace filesystem markers with DuckDB
- **Async Data Loading** - Non-blocking I/O for dashboard
- **GitHub Actions** - CI/CD pipeline
- **Requirements Pinning** - Lock all transitive dependencies

### Medium Priority (Weeks 3-4)

- Stacking ensemble metadata tracking
- Feature importance analysis
- API rate limiting
- Dashboard real-time updates
- Backtesting framework
- Custom loss functions

---

## Performance Considerations

### Data Processing
- **Ingestion:** ~10s for 5-year history (yfinance API)
- **Preprocessing:** ~2s for single equity
- **Feature Generation:** ~5s for 80 features
- **Bottleneck:** yfinance API rate limiting

### Model Training
- **LSTM:** 2-5 minutes (5 years of data, GPU)
- **XGBoost:** 10-30 seconds
- **LightGBM:** 5-15 seconds
- **Hyperparameter Optimization:** 10-60 minutes (100 trials)
- **Walk-Forward Validation:** 5-15 minutes (10 folds)

**Hardware Requirements:**
- GPU: NVIDIA (CUDA 11.8+) recommended for deep learning
- RAM: 8-16GB minimum, 32GB recommended
- Storage: 500GB+ for model cache and data

### Inference
- **Single Prediction:** <100ms (excluding feature preparation)
- **Batch (100 symbols):** <5s
- **Dashboard Latency:** 1-2s (API + rendering)

### Caching Strategy
- **Features:** Cached per symbol, invalidated daily
- **Models:** Loaded once, reused for batch predictions
- **OOF Predictions:** Generated once per training cycle

### Optimization Opportunities
1. **Parallel Training:** Train models in parallel across symbols
2. **Feature Caching:** Pre-compute features for hot symbols
3. **Async I/O:** Non-blocking API calls in dashboard
4. **Model Quantization:** Reduce model size 4-10x
5. **Batch Inference:** Vectorize predictions across symbols

---

## Contributing & Maintenance

### Adding a New Model

**1. Implement Model Class**
```python
# src/models/new_model.py
from models.base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = ...
    
    def train(self, X_train, y_train, X_val, y_val):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

**2. Register in Pipeline**
```python
# src/pipeline/E_modeltrainer_pipeline.py
modelconfigs = {
    'new_model': NewModel(),
    ...
}
```

**3. Add Tests**
```python
# tests/models/test_new_model.py
def test_train():
    model = NewModel()
    # assertions
```

### Adding a New Pipeline Stage

**1. Create Pipeline File**
```python
# src/pipeline/I_new_stage_pipeline.py
def run(config: FullConfig) -> dict:
    # Implementation
    return results
```

**2. Add to DAG**
```python
# src/dag/dag_config.py
dag.add_node('I_new_stage', I_new_stage_pipeline.run)
dag.add_edge('H_ensemble', 'I_new_stage')
```

### Testing Strategy

**Unit Tests:** Test individual functions
```
tests/models/test_lstm_model.py
tests/validation/test_validator.py
```

**Integration Tests:** Test pipeline end-to-end
```
tests/test_integration_e2e.py
```

**Run Tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=src/  # With coverage
```

---

## Future Roadmap

### Phase 1 (Critical - Weeks 1-3)
✅ Data validation framework
✅ Type checking
✅ API input validation
✅ Error logging

### Phase 2 (High - Weeks 4-7)
- MLflow model versioning
- Structured JSON logging
- Database-backed state
- Async feature loading
- GitHub Actions CI/CD

### Phase 3 (Medium - Weeks 8-12)
- Stacking ensemble improvements
- Feature importance tracking
- Advanced backtesting
- Real-time dashboard
- AWS Lambda integration

### Phase 4 (Nice-to-have - Weeks 13+)
- Multi-timeframe forecasting
- Sentiment-aware retraining
- Portfolio optimization
- Risk management module
- Advanced monitoring

---

## Contact & Support

For questions or issues:
- See `IMPLEMENTATION_ROADMAP.md` for technical details
- See `FILE_MODIFICATION_GUIDE.md` for specific code changes
- See `TRACKING_CHECKLIST.md` for progress tracking

---
  
**Next Review:** After Phase 1 completion (AWS readiness checkpoint)
