# Notebooks

Exploratory companions to the pipeline. All four run fully offline on the
synthetic market generator — no data downloads, no API keys.

| Notebook | Contents |
|---|---|
| `01_data_exploration.ipynb` | Synthetic regime-switching market, data quality report, volatility clustering |
| `02_feature_research.ipynb` | Causal feature panel, HMM stress probability vs realized vol, feature correlation |
| `03_model_comparison.ipynb` | Walk-forward model comparison, Newey-West IC inference, PBO |
| `04_backtest_analysis.ipynb` | Signals → portfolio → costed backtest → deflated Sharpe |

The production pipeline lives in `scripts/` and `alphaforge/` so experiments
remain reproducible outside notebooks; notebooks call the same library code.
