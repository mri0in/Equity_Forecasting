print("""
User Input (Dashboard)
          │
          ▼
  set_active_equity()
          │
          ▼
    ACTIVE_EQUITY  (global state)
       ┌───────┴────────┐
       │                │
  Market Sentiment   Forecasting
  ┌─────────────┐   ┌──────────────┐
  │Feeds → APIs │   │Historical    │
  │Preprocess   │   │Data Loader   │
  │Extractor    │   │Forecast Model│
  │Aggregator   │   └──────────────┘
  └─────────────┘
       │
       ▼
Combined Output → Dashboard
""")
