"""
Experiment tracking system for reproducible research.

Tracks model configurations, metrics, and results for easy comparison.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Simple experiment tracking for machine learning experiments.

    Manages experiment metadata, configurations, and results.
    Enables easy comparison across different model runs.
    """

    def __init__(self, base_dir: str | Path = 'experiments'):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        base_dir : str or Path, default='experiments'
            Base directory for storing experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        # Master log file
        self.master_log = self.base_dir / 'experiment_log.json'
        self._load_master_log()

    def _load_master_log(self):
        """Load or initialize master experiment log."""
        if self.master_log.exists():
            with open(self.master_log, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []

    def _save_master_log(self):
        """Save master experiment log."""
        with open(self.master_log, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def start_experiment(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> Path:
        """
        Start a new experiment and create its directory.

        Parameters
        ----------
        name : str
            Experiment name (e.g., 'tfa_baseline', 'ridge_alpha_tuning')
        description : str
            Brief description of the experiment
        config : dict
            Configuration dictionary (hyperparameters, etc.)
        tags : list of str, optional
            Tags for categorization (e.g., ['baseline', 'transformer'])

        Returns
        -------
        exp_dir : Path
            Path to experiment directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{timestamp}_{name}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)

        # Create experiment metadata
        metadata = {
            'id': exp_id,
            'name': name,
            'timestamp': timestamp,
            'description': description,
            'tags': tags or [],
            'status': 'running',
        }

        # Save configuration
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Create notes file
        with open(exp_dir / 'notes.md', 'w') as f:
            f.write(f"# {name}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Description**: {description}\n\n")
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(config, indent=2))
            f.write("\n```\n\n")
            f.write("## Observations\n\n")
            f.write("- \n\n")
            f.write("## Results\n\n")
            f.write("(To be filled after training)\n\n")

        # Add to master log
        self.experiments.append(metadata)
        self._save_master_log()

        logger.info(f"Started experiment: {exp_id}")
        logger.info(f"  Directory: {exp_dir}")
        logger.info(f"  Description: {description}")

        return exp_dir

    def log_metrics(
        self,
        exp_dir: Path | str,
        metrics: Dict[str, float],
        stage: str = 'final',
    ):
        """
        Log metrics for an experiment.

        Parameters
        ----------
        exp_dir : Path or str
            Experiment directory
        metrics : dict
            Dictionary of metric names and values
        stage : str, default='final'
            Stage identifier (e.g., 'final', 'epoch_10', 'validation')
        """
        exp_dir = Path(exp_dir)
        metrics_file = exp_dir / f'metrics_{stage}.json'

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Logged metrics for {exp_dir.name} ({stage}):")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")

    def finish_experiment(
        self,
        exp_dir: Path | str,
        status: str = 'completed',
        summary: Optional[str] = None,
    ):
        """
        Mark experiment as finished.

        Parameters
        ----------
        exp_dir : Path or str
            Experiment directory
        status : str, default='completed'
            Final status ('completed', 'failed', 'cancelled')
        summary : str, optional
            Brief summary of results
        """
        exp_dir = Path(exp_dir)
        exp_id = exp_dir.name

        # Update master log
        for exp in self.experiments:
            if exp['id'] == exp_id:
                exp['status'] = status
                exp['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if summary:
                    exp['summary'] = summary
                break

        self._save_master_log()

        logger.info(f"Experiment {exp_id} marked as {status}")

    def compare_experiments(
        self,
        exp_ids: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        metric_stage: str = 'final',
    ) -> pd.DataFrame:
        """
        Compare metrics across experiments.

        Parameters
        ----------
        exp_ids : list of str, optional
            Specific experiment IDs to compare. If None, compare all.
        tags : list of str, optional
            Filter experiments by tags
        metric_stage : str, default='final'
            Which metrics file to load (e.g., 'final', 'validation')

        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame with experiment comparisons
        """
        results = []

        # Filter experiments
        experiments = self.experiments
        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.get('tags', []) for tag in tags)
            ]
        if exp_ids:
            experiments = [
                exp for exp in experiments
                if exp['id'] in exp_ids
            ]

        for exp in experiments:
            exp_dir = self.base_dir / exp['id']
            metrics_file = exp_dir / f'metrics_{metric_stage}.json'

            if not metrics_file.exists():
                logger.warning(f"No metrics found for {exp['id']}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            row = {
                'experiment_id': exp['id'],
                'name': exp['name'],
                'status': exp.get('status', 'unknown'),
                'timestamp': exp.get('timestamp', ''),
                **metrics
            }
            results.append(row)

        if len(results) == 0:
            logger.warning("No experiments found for comparison")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Save comparison
        comparison_file = self.base_dir / f'comparison_{metric_stage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(comparison_file, index=False)
        logger.info(f"Comparison saved to {comparison_file}")

        return df

    def get_best_experiment(
        self,
        metric: str,
        maximize: bool = True,
        tags: Optional[list[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find best experiment based on a metric.

        Parameters
        ----------
        metric : str
            Metric name to optimize
        maximize : bool, default=True
            Whether to maximize (True) or minimize (False) the metric
        tags : list of str, optional
            Filter by tags

        Returns
        -------
        best_exp : dict or None
            Best experiment metadata and metrics
        """
        df = self.compare_experiments(tags=tags)

        if df.empty or metric not in df.columns:
            logger.warning(f"No experiments found with metric '{metric}'")
            return None

        if maximize:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()

        best_row = df.loc[best_idx]
        logger.info(f"Best experiment for {metric}: {best_row['name']}")
        logger.info(f"  {metric}: {best_row[metric]:.6f}")

        return best_row.to_dict()

    def list_experiments(
        self,
        tags: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        List all experiments with optional filtering.

        Parameters
        ----------
        tags : list of str, optional
            Filter by tags
        status : str, optional
            Filter by status ('running', 'completed', 'failed')

        Returns
        -------
        df : pd.DataFrame
            DataFrame of experiments
        """
        experiments = self.experiments

        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.get('tags', []) for tag in tags)
            ]

        if status:
            experiments = [
                exp for exp in experiments
                if exp.get('status') == status
            ]

        df = pd.DataFrame(experiments)

        if not df.empty:
            # Select and order columns
            columns = ['id', 'name', 'timestamp', 'status', 'description']
            columns = [col for col in columns if col in df.columns]
            df = df[columns]

        return df

    def export_to_latex(
        self,
        comparison_df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        caption: str = "Experiment Results",
        label: str = "tab:results",
    ) -> str:
        """
        Export comparison to LaTeX table.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Comparison dataframe from compare_experiments()
        columns : list of str, optional
            Columns to include. If None, include all numeric columns.
        caption : str
            Table caption
        label : str
            LaTeX label for referencing

        Returns
        -------
        latex_str : str
            LaTeX table string
        """
        if columns is None:
            # Include name and all numeric columns
            columns = ['name'] + [
                col for col in comparison_df.columns
                if pd.api.types.is_numeric_dtype(comparison_df[col])
            ]

        df_subset = comparison_df[columns].copy()

        # Format numeric columns
        for col in df_subset.columns:
            if pd.api.types.is_numeric_dtype(df_subset[col]):
                df_subset[col] = df_subset[col].apply(lambda x: f"{x:.4f}")

        latex_str = df_subset.to_latex(
            index=False,
            caption=caption,
            label=label,
            escape=False,
        )

        return latex_str
