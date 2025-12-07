"""
Alpha-Hunter: Dynamic Factor Investing with Transformer-Based Return Prediction

Core modules for model training, evaluation, and backtesting.
"""

__version__ = "0.2.0"

from .models import (
    TransformerPredictor,
    RidgePredictor,
    RandomForestPredictor,
    MLPPredictor,
)
from .models_tfa import (
    TemporalFactorAutoencoder,
    TFAPredictor,
)
from .data_loader import SequenceDataLoader
from .trainer import RollingWindowTrainer
from .evaluator import PerformanceEvaluator
from .tfa_analysis import TFAAnalyzer
from .experiment_tracker import ExperimentTracker

__all__ = [
    "TransformerPredictor",
    "RidgePredictor",
    "RandomForestPredictor",
    "MLPPredictor",
    "TemporalFactorAutoencoder",
    "TFAPredictor",
    "SequenceDataLoader",
    "RollingWindowTrainer",
    "PerformanceEvaluator",
    "TFAAnalyzer",
    "ExperimentTracker",
]

