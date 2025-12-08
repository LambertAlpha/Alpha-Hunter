"""
Command-line training script for Alpha-Hunter models.

Usage:
    python train.py --model transformer --config config.json
    python train.py --model ridge
    python train.py --model all  # Train all models
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import SequenceDataLoader
from src.models import (
    TransformerPredictor,
    RidgePredictor,
    RandomForestPredictor,
    MLPPredictor,
)
from src.trainer import RollingWindowTrainer
from src.evaluator import PerformanceEvaluator
from src.config import Config
from src.utils import (
    setup_logger,
    set_random_seed,
    check_gpu_availability,
    ensure_dir,
    create_timestamp,
)


def get_model_factory(model_name: str, config: Config, device: str = 'cpu', data_loader=None):
    """
    Get model factory function.
    
    Parameters
    ----------
    model_name : str
        Model name: 'transformer', 'ridge', 'random_forest', 'mlp'
    config : Config
        Configuration object
    device : str
        Device for PyTorch models
    data_loader : SequenceDataLoader, optional
        Data loader for getting feature dimensions
        
    Returns
    -------
    factory : callable
        Function that returns a new model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'transformer':
        def factory():
            # Get actual input_dim from data_loader stats if available
            if data_loader is not None:
                stats = data_loader.get_statistics()
                input_dim = stats['n_features']
            else:
                input_dim = 11  # Default fallback
            return TransformerPredictor(
                input_dim=input_dim,
                d_model=config.transformer.d_model,
                nhead=config.transformer.nhead,
                num_layers=config.transformer.num_layers,
                dim_feedforward=config.transformer.dim_feedforward,
                dropout=config.transformer.dropout,
                lr=config.transformer.lr,
                weight_decay=config.transformer.weight_decay,
                batch_size=config.transformer.batch_size,
                epochs=config.transformer.epochs,
                early_stopping_patience=config.transformer.early_stopping_patience,
                device=device,
            )
    
    elif model_name == 'ridge':
        def factory():
            return RidgePredictor(alpha=config.ridge.alpha)
    
    elif model_name == 'random_forest' or model_name == 'rf':
        def factory():
            return RandomForestPredictor(
                n_estimators=config.random_forest.n_estimators,
                max_depth=config.random_forest.max_depth,
                min_samples_split=config.random_forest.min_samples_split,
                n_jobs=config.random_forest.n_jobs,
                random_state=config.random_forest.random_state,
            )
    
    elif model_name == 'mlp':
        def factory():
            # Dynamic input dimension: n_features * sequence_length (use actual n_features from data_loader if available)
            input_dim = (
                data_loader.get_statistics()['n_features'] * config.data.sequence_length
                if data_loader is not None else 11 * config.data.sequence_length
            )
            return MLPPredictor(
                input_dim=input_dim,
                hidden_dims=config.mlp.hidden_dims,
                dropout=config.mlp.dropout,
                lr=config.mlp.lr,
                weight_decay=config.mlp.weight_decay,
                batch_size=config.mlp.batch_size,
                epochs=config.mlp.epochs,
                device=device,
            )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return factory


def train_model(
    model_name: str,
    data_loader: SequenceDataLoader,
    config: Config,
    output_dir: Path,
    device: str = 'cpu',
    logger=None,
    max_prediction_dates: Optional[int] = None,
    prediction_step: int = 1,
):
    """
    Train a single model.
    
    Parameters
    ----------
    model_name : str
        Model name
    data_loader : SequenceDataLoader
        Data loader
    config : Config
        Configuration
    output_dir : Path
        Output directory
    device : str
        Device for PyTorch models
    logger : logging.Logger
        Logger instance
    """
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()} model")
        logger.info(f"{'='*60}")
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    ensure_dir(model_output_dir)
    
    # Get model factory
    factory = get_model_factory(model_name, config, device, data_loader)
    
    # Create trainer
    trainer = RollingWindowTrainer(
        data_loader=data_loader,
        model_factory=factory,
        train_window=config.training.train_window,
        val_window=config.training.val_window,
        min_train_months=config.training.min_train_months,
        output_dir=model_output_dir,
    )
    
    # Train and predict
    predictions_df = trainer.train_and_predict(
        save_models=config.training.save_models,
        verbose=config.training.verbose,
        max_prediction_dates=max_prediction_dates,
        prediction_step=prediction_step,
    )
    
    if len(predictions_df) == 0:
        if logger:
            logger.warning(f"No predictions generated for {model_name}")
        return None
    
    # Evaluate
    evaluator = PerformanceEvaluator()
    
    # Compute portfolio returns
    portfolio_df = evaluator.compute_portfolio_returns(
        predictions_df,
        long_pct=config.evaluation.long_pct,
        short_pct=config.evaluation.short_pct,
        transaction_cost=config.evaluation.transaction_cost,
    )
    
    # Compute summary statistics
    stats = evaluator.compute_summary_statistics(predictions_df, portfolio_df)
    
    # Print summary
    evaluator.print_summary(stats)
    
    # Plot performance
    ic_series = evaluator.compute_ic(predictions_df)
    evaluator.plot_performance(
        ic_series,
        portfolio_df,
        save_path=model_output_dir / f'performance_{create_timestamp()}.png',
    )
    
    # Save results
    predictions_df.to_csv(model_output_dir / f'predictions_{create_timestamp()}.csv', index=False)
    portfolio_df.to_csv(model_output_dir / f'portfolio_{create_timestamp()}.csv')
    
    import json
    with open(model_output_dir / f'stats_{create_timestamp()}.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    if logger:
        logger.info(f"Results saved to {model_output_dir}")
    
    return stats


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Alpha-Hunter models')
    
    parser.add_argument(
        '--model',
        type=str,
        default='transformer',
        choices=['transformer', 'ridge', 'random_forest', 'rf', 'mlp', 'all'],
        help='Model to train (default: transformer)',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file (optional)',
    )
    
    parser.add_argument(
        '--pca_path',
        type=str,
        default='feature/pca_feature_store.csv',
        help='Path to PCA features CSV',
    )
    parser.add_argument(
        '--returns_path',
        type=str,
        default='data/cleaned/monthly_returns_proxy.csv',
        help='Path to returns CSV (date, asset, return). Set to blank to skip merge.',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for PyTorch models (auto will detect GPU: CUDA/MPS)',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output',
    )
    
    parser.add_argument(
        '--max_prediction_dates',
        type=int,
        default=None,
        help='Max number of prediction dates (for faster testing)',
    )
    
    parser.add_argument(
        '--prediction_step',
        type=int,
        default=1,
        help='Step size for prediction dates (1=every date, 2=every other, etc.)',
    )
    
    args = parser.parse_args()
    
    # Setup
    set_random_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    
    # Setup logger
    log_file = output_dir / f'train_{create_timestamp()}.log'
    logger = setup_logger('alpha_hunter', log_file=str(log_file))
    
    logger.info("="*60)
    logger.info("Alpha-Hunter Training Pipeline")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"PCA Path: {args.pca_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Random Seed: {args.seed}")
    
    # Device selection
    if args.device == 'auto':
        device = check_gpu_availability()
    else:
        device = args.device
    logger.info(f"Device: {device}")
    
    # Load configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = Config.load(args.config)
    else:
        logger.info("Using default configuration")
        config = Config()
    
    # Override config with command-line args
    config.data.pca_path = args.pca_path
    config.data.returns_path = args.returns_path
    config.training.output_dir = str(output_dir)
    config.training.verbose = args.verbose
    
    # Save configuration
    config.save(output_dir / 'config.json')
    logger.info(f"Configuration saved to {output_dir / 'config.json'}")
    
    # Load data
    logger.info(f"\nLoading data from {args.pca_path}")
    data_loader = SequenceDataLoader(
        pca_path=config.data.pca_path,
        returns_path=config.data.returns_path if config.data.returns_path else None,
        sequence_length=config.data.sequence_length,
        forward_fill_limit=config.data.forward_fill_limit,
    )
    
    # Print dataset statistics
    stats = data_loader.get_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Dates: {stats['n_dates']}")
    logger.info(f"  Assets: {stats['n_assets']}")
    logger.info(f"  Features: {stats['n_features']}")
    logger.info(f"  Date Range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
    logger.info(f"  Avg Assets/Date: {stats['avg_assets_per_date']:.1f}")
    
    # Train model(s)
    if args.model == 'all':
        models = ['ridge', 'random_forest', 'mlp', 'transformer']
        results = {}
        
        for model_name in models:
            try:
                stats = train_model(
                    model_name,
                    data_loader,
                    config,
                    output_dir,
                    device,
                    logger,
                    max_prediction_dates=args.max_prediction_dates,
                    prediction_step=args.prediction_step,
                )
                results[model_name] = stats
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Compare results
        logger.info(f"\n{'='*60}")
        logger.info("Model Comparison")
        logger.info(f"{'='*60}")
        
        comparison_df = pd.DataFrame(results).T
        logger.info(f"\n{comparison_df.to_string()}")
        
        comparison_df.to_csv(output_dir / f'model_comparison_{create_timestamp()}.csv')
        logger.info(f"\nComparison saved to {output_dir / 'model_comparison.csv'}")
        
    else:
        train_model(
            args.model,
            data_loader,
            config,
            output_dir,
            device,
            logger,
            max_prediction_dates=args.max_prediction_dates,
            prediction_step=args.prediction_step,
        )
    
    logger.info(f"\n{'='*60}")
    logger.info("Training completed successfully!")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    import pandas as pd  # noqa: F401
    main()
