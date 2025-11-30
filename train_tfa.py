"""
Training script for Temporal Factor Autoencoder (TFA).

Usage:
    python train_tfa.py --config config.json
    python train_tfa.py --epochs 100 --batch_size 256
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import SequenceDataLoader
from src.models_tfa import TFAPredictor
from src.trainer import RollingWindowTrainer
from src.evaluator import PerformanceEvaluator
from src.tfa_analysis import TFAAnalyzer
from src.config import Config, TFAConfig
from src.utils import (
    setup_logger,
    set_random_seed,
    check_gpu_availability,
    ensure_dir,
    create_timestamp,
)


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Factor Autoencoder')
    
    # Data arguments
    parser.add_argument(
        '--pca_path',
        type=str,
        default='feature/pca_feature_store.csv',
        help='Path to PCA features'
    )
    
    # Model arguments  
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=4, help='Encoder layers')
    parser.add_argument('--n_latent_factors', type=int, default=5, help='Latent factors')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.1, help='Reconstruction weight')
    parser.add_argument('--beta', type=float, default=0.05, help='Smoothness weight')
    parser.add_argument('--gamma', type=float, default=0.01, help='Orthogonality weight')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/tfa', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after training')
    
    args = parser.parse_args()
    
    # Setup
    set_random_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    
    # Device
    if args.device == 'auto':
        device = check_gpu_availability()
    else:
        device = args.device
    
    # Logger
    log_file = output_dir / f'train_tfa_{create_timestamp()}.log'
    logger = setup_logger('tfa_training', log_file=str(log_file))
    
    logger.info("="*60)
    logger.info("Temporal Factor Autoencoder Training")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"PCA Path: {args.pca_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load configuration
    config = Config()
    
    # Override with command line arguments
    config.tfa.d_model = args.d_model
    config.tfa.n_heads = args.n_heads
    config.tfa.n_encoder_layers = args.n_encoder_layers
    config.tfa.n_latent_factors = args.n_latent_factors
    config.tfa.epochs = args.epochs
    config.tfa.batch_size = args.batch_size
    config.tfa.lr = args.lr
    config.tfa.alpha = args.alpha
    config.tfa.beta = args.beta
    config.tfa.gamma = args.gamma
    config.tfa.device = device
    
    # Save config
    config.save(output_dir / 'config.json')
    logger.info(f"Configuration saved")
    
    # Load data
    logger.info(f"\nLoading data from {args.pca_path}")
    data_loader = SequenceDataLoader(
        pca_path=args.pca_path,
        sequence_length=config.tfa.seq_len,  # 36 months
        forward_fill_limit=config.data.forward_fill_limit,
    )
    
    stats = data_loader.get_statistics()
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Dates: {stats['n_dates']}")
    logger.info(f"  Assets: {stats['n_assets']}")
    logger.info(f"  Features: {stats['n_features']}")
    logger.info(f"  Date Range: {stats['date_range']}")
    
    # Create model factory
    def create_tfa():
        return TFAPredictor(
            n_pca_factors=stats['n_features'],
            seq_len=config.tfa.seq_len,
            d_model=config.tfa.d_model,
            n_heads=config.tfa.n_heads,
            n_encoder_layers=config.tfa.n_encoder_layers,
            n_decoder_layers=config.tfa.n_decoder_layers,
            n_latent_factors=config.tfa.n_latent_factors,
            dropout=config.tfa.dropout,
            n_classes=config.tfa.n_classes,
            lr=config.tfa.lr,
            weight_decay=config.tfa.weight_decay,
            batch_size=config.tfa.batch_size,
            epochs=config.tfa.epochs,
            early_stopping_patience=config.tfa.early_stopping_patience,
            alpha=config.tfa.alpha,
            beta=config.tfa.beta,
            gamma=config.tfa.gamma,
            device=device,
        )
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = RollingWindowTrainer(
        data_loader=data_loader,
        model_factory=create_tfa,
        train_window=config.training.train_window,
        val_window=config.training.val_window,
        min_train_months=config.training.min_train_months,
        output_dir=output_dir,
    )
    
    # Train and predict
    logger.info("\nStarting rolling window training...")
    logger.info(f"Train window: {config.training.train_window} months")
    logger.info(f"Val window: {config.training.val_window} months")
    
    predictions_df = trainer.train_and_predict(
        save_models=False,
        verbose=args.verbose,
    )
    
    if len(predictions_df) == 0:
        logger.error("No predictions generated!")
        return
    
    logger.info(f"\n✅ Generated {len(predictions_df)} predictions")
    
    # Evaluate
    logger.info("\nEvaluating performance...")
    evaluator = PerformanceEvaluator()
    
    # Compute metrics
    ic_series = evaluator.compute_ic(predictions_df)
    icir = evaluator.compute_icir(predictions_df)
    
    portfolio_df = evaluator.compute_portfolio_returns(
        predictions_df,
        long_pct=config.evaluation.long_pct,
        short_pct=config.evaluation.short_pct,
        transaction_cost=config.evaluation.transaction_cost,
    )
    
    sharpe = evaluator.compute_sharpe_ratio(portfolio_df, column='ls_ret_net')
    
    # Summary statistics
    stats = evaluator.compute_summary_statistics(predictions_df, portfolio_df)
    
    logger.info("\n" + "="*60)
    logger.info("Performance Summary")
    logger.info("="*60)
    evaluator.print_summary(stats)
    
    # Plot
    evaluator.plot_performance(
        ic_series,
        portfolio_df,
        save_path=output_dir / f'performance_{create_timestamp()}.png'
    )
    
    # Save results
    predictions_df.to_csv(output_dir / f'predictions_{create_timestamp()}.csv', index=False)
    portfolio_df.to_csv(output_dir / f'portfolio_{create_timestamp()}.csv')
    
    import json
    with open(output_dir / f'stats_{create_timestamp()}.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_dir}")
    
    # Analysis
    if args.analyze:
        logger.info("\n" + "="*60)
        logger.info("Running TFA Analysis")
        logger.info("="*60)
        
        # Load a trained model (use the last one)
        if hasattr(trainer, 'models') and len(trainer.models) > 0:
            last_date = max(trainer.models.keys())
            model = trainer.models[last_date]
            
            analyzer = TFAAnalyzer(model, device=device)
            
            # Get test data
            test_date = predictions_df['date'].max()
            test_data_dict = data_loader.build_sequences(
                target_date=test_date,
                include_target=True,
                return_dict=True
            )
            
            X_test = test_data_dict['X']
            y_test = test_data_dict['y']
            
            # Generate report
            analysis_dir = output_dir / 'analysis'
            analyzer.generate_report(
                X_test,
                y_test,
                dates=pd.DatetimeIndex([test_date] * len(X_test)),
                output_dir=str(analysis_dir)
            )
            
            logger.info(f"\n✅ Analysis saved to {analysis_dir}")
        else:
            logger.warning("No trained models available for analysis")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info("="*60)


if __name__ == '__main__':
    main()

