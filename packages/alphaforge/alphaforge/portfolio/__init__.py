"""Portfolio construction: legacy signal pipeline plus the SF-S4-MR1 contracts."""

from alphaforge.portfolio.allocation import (
    POLICIES,
    apply_uncertainty_sizing,
    apply_volatility_target,
    inverse_volatility_portfolio,
    long_short_spread_portfolio,
    quantile_portfolio,
    rank_weighted_portfolio,
    score_weighted_portfolio,
    top_k_portfolio,
)
from alphaforge.portfolio.construction import construct_portfolio
from alphaforge.portfolio.contracts import (
    AllocationResult,
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
    project_to_feasible,
)
from alphaforge.portfolio.evidence import (
    BacktestPanel,
    Fold,
    capacity_frontier,
    chronological_folds,
    compare_allocation_policies,
    run_allocation_backtest,
    score_final_holdout,
    summarize_record,
    volatility_regimes,
)

__all__ = [
    "POLICIES",
    "AllocationResult",
    "BacktestPanel",
    "Fold",
    "InfeasibleConstraintsError",
    "PortfolioConstraints",
    "PortfolioError",
    "apply_uncertainty_sizing",
    "apply_volatility_target",
    "capacity_frontier",
    "chronological_folds",
    "compare_allocation_policies",
    "construct_portfolio",
    "inverse_volatility_portfolio",
    "liquidity_caps_from_adv",
    "long_short_spread_portfolio",
    "project_to_feasible",
    "quantile_portfolio",
    "rank_weighted_portfolio",
    "run_allocation_backtest",
    "score_final_holdout",
    "score_weighted_portfolio",
    "summarize_record",
    "top_k_portfolio",
    "volatility_regimes",
]
