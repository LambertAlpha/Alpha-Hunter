"""Build a frozen PCA basis from historical calibration data; never fill return labels.

Inputs are long-form CSVs. Cleaned features must already be available as of their
stated month. This script cannot certify upstream cleaning or data vendor rights.
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.cli import sha256_file, write_json


def month(value) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if not isinstance(date, pd.Timestamp) or pd.isna(date):
        raise ValueError('Expected a valid month')
    return date.to_period('M').to_timestamp()


def read_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'asset': str})
    if frame.empty or not {'date', 'asset'}.issubset(frame.columns):
        raise ValueError('Nonempty panel requires date and asset columns')
    if frame[['date', 'asset']].isna().to_numpy().any() or frame.asset.str.strip().eq('').any():
        raise ValueError('Missing date/asset identifiers')
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.to_period('M').dt.to_timestamp()
    if frame.date.isna().any() or frame.duplicated(['date', 'asset']).any():
        raise ValueError('Missing date or duplicate asset/month')
    return frame.sort_values(['date', 'asset']).reset_index(drop=True)


def frozen_pca(features: pd.DataFrame, calibration_start, calibration_end, feature_end,
               max_components: int = 10, variance_target: float = .8):
    """Fit once; export only months after calibration, in one unchanging basis.

    No extra winsorization, per-month neutralization or scaling is introduced.
    Centering is learned exclusively from the calibration rows. Full SVD makes
    the requested variance threshold measurable even when the hard cap binds.
    """
    start, end, last = map(month, [calibration_start, calibration_end, feature_end])
    if not start <= end < last:
        raise ValueError('Require calibration_start <= calibration_end < feature_end')
    if not isinstance(max_components, int) or max_components < 1 or not 0 < variance_target <= 1:
        raise ValueError('Positive component cap and variance target in (0, 1] required')
    columns = [c for c in features if c not in ['date', 'asset']]
    if not columns or {'return', 'target', 'label'}.intersection(columns):
        raise ValueError('Expected feature columns without return/target/label columns')
    dates = pd.DatetimeIndex(sorted(features.date.unique()))
    required = pd.date_range(start, last, freq='MS')
    if not required.isin(dates).all():
        raise ValueError('Missing required feature calendar months')
    # Future rows beyond feature_end cannot affect this build or its validation.
    frame = pd.DataFrame(features.loc[features.date.between(start, last)]).copy()
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite; resolve missingness upstream')
    fit_mask = frame.date.le(end).to_numpy()
    training = values[fit_mask]
    if len(training) < 2 or np.var(training, axis=0).sum() <= 0:
        raise ValueError('Calibration must contain nonconstant features')
    pca = PCA(svd_solver='full').fit(training)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    required_components = min(int(np.searchsorted(cumulative, variance_target)) + 1, len(cumulative))
    count = min(max_components, required_components, len(cumulative))
    assert pca.components_ is not None and pca.mean_ is not None
    components = pca.components_[:count]
    projected = (values[~fit_mask] - pca.mean_) @ components.T
    scores = frame.loc[~fit_mask, ['date', 'asset']].reset_index(drop=True)
    names = [f'pca_{i + 1}' for i in range(count)]
    scores[names] = projected
    explained = float(cumulative[count - 1])
    metadata: dict[str, Any] = dict(method='frozen full-SVD PCA; no added neutralization or standardization',
                    calibration_start=str(start.date()), calibration_end=str(end.date()),
                    feature_start=str(scores.date.min().date()), feature_end=str(last.date()),
                    calibration_rows=len(training), feature_rows=len(scores), feature_count=len(columns),
                    feature_names=columns, pca_names=names, max_components=max_components,
                    components=count, variance_target=variance_target,
                    explained_variance=explained, variance_target_met=explained >= variance_target,
                    components_needed_for_target=required_components,
                    explained_variance_ratio=pca.explained_variance_ratio_[:count].tolist())
    return scores, metadata, dict(components=components, mean=pca.mean_, feature_names=np.array(columns))


def label_coverage(features: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Count labels for last-feature-month members, before any history filter."""
    expected = features[['date', 'asset']].copy()
    expected['date'] = expected.date + pd.offsets.MonthBegin(1)
    coverage = expected.merge(returns[['date', 'asset', 'return']], on=['date', 'asset'],
                              how='left', validate='one_to_one')
    return coverage.groupby('date')['return'].agg(eligible_members='size', observed_labels='count').assign(
        missing_labels=lambda x: x.eligible_members - x.observed_labels).reset_index()


def prepare(features_path: Path, returns_path: Path, output: Path, calibration_start: str,
            calibration_end: str, feature_end: str, max_components: int = 10, variance_target: float = .8):
    if output.exists():
        raise FileExistsError(f'Output directory already exists: {output}')
    features, returns = read_panel(features_path), read_panel(returns_path)
    if 'return' not in returns:
        raise ValueError('Returns panel requires a return column')
    returns['return'] = pd.to_numeric(returns['return'], errors='raise')
    return_values = returns['return'].to_numpy(dtype=float)
    if np.isinf(return_values).any() or (return_values < -1).any():
        raise ValueError('Simple returns must be finite or missing and >= -1')
    scores, metadata, basis = frozen_pca(features, calibration_start, calibration_end, feature_end,
                                        max_components, variance_target)
    returns = pd.DataFrame(returns.loc[
        returns.date.between(scores.date.min(), month(feature_end) + pd.offsets.MonthBegin(1))
        & returns.asset.isin(scores.asset.unique())]).reindex(columns=['date', 'asset', 'return'])
    coverage = label_coverage(scores, returns)
    metadata.update(kind='retrospective diagnostic; upstream point-in-time provenance unverified',
                    assumptions=['Input features were available by their month end',
                                 'Return labels are decimal simple returns earned during the stated month',
                                 'Feature membership is an eligibility proxy, not verified tradability'],
                    inputs={name: dict(filename=path.name, sha256=sha256_file(path)) for name, path in
                            [('features', features_path), ('returns', returns_path)]},
                    script_sha256=sha256_file(Path(__file__)),
                    missing_next_month_labels=int(coverage.missing_labels.sum()))
    output.mkdir(parents=True)
    scores.to_csv(output / 'pca_features.csv', index=False)
    returns.to_csv(output / 'returns.csv', index=False)
    coverage.to_csv(output / 'label_coverage.csv', index=False)
    np.savez(output / 'pca_basis.npz', components=basis['components'], mean=basis['mean'],
             feature_names=basis['feature_names'])
    metadata['outputs_sha256'] = {p.name: sha256_file(p) for p in sorted(output.iterdir())}
    write_json(output / 'preparation.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--returns', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-start', required=True)
    parser.add_argument('--calibration-end', required=True)
    parser.add_argument('--feature-end', required=True)
    parser.add_argument('--max-components', type=int, default=10)
    parser.add_argument('--variance-target', type=float, default=.8)
    args = parser.parse_args()
    result = prepare(args.features, args.returns, args.output, args.calibration_start, args.calibration_end,
                     args.feature_end, args.max_components, args.variance_target)
    print(f"Prepared {result['feature_rows']} rows, {result['components']} components; "
          f"variance={result['explained_variance']:.4f}, target_met={result['variance_target_met']}")


if __name__ == '__main__':
    main()
