"""RetailPulse Machine Learning & Modeling Package"""

from .segmentation import (
    assign_rfm_quintiles,
    find_optimal_k,
    train_customer_segmentation
)
from .prophet_model import (
    prepare_prophet_data,
    train_prophet,
    compute_forecast_metrics
)
from .lstm_model import (
    create_sequences,
    train_lstm
)
from .hybrid import (
    train_hybrid
)
from .churn import (
    prepare_churn_data,
    train_churn_models,
    CHURN_FEATURE_COLS
)
from .tune import (
    tune_churn_model
)

__all__ = [
    "assign_rfm_quintiles",
    "find_optimal_k",
    "train_customer_segmentation",
    "prepare_prophet_data",
    "train_prophet",
    "compute_forecast_metrics",
    "create_sequences",
    "train_lstm",
    "train_hybrid",
    "prepare_churn_data",
    "train_churn_models",
    "CHURN_FEATURE_COLS",
    "tune_churn_model"
]
