"""Hand-calculated portfolio accounting regressions."""
import unittest

import numpy as np
import pandas as pd

from src.evaluator import PerformanceEvaluator


def panel(returns=((0.1, -0.1),)):
    return pd.DataFrame([
        dict(date=date, asset=asset, prediction=score, actual_return=ret)
        for date, row in zip(pd.date_range('2020-01-01', periods=len(returns), freq='MS'), returns)
        for asset, score, ret in zip(['A', 'B'], [1., 0.], row)
    ])


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.e = PerformanceEvaluator()

    def test_opening_cost_is_charged_once(self):
        result = self.e.compute_portfolio_returns(panel(), transaction_cost=0.001)
        self.assertAlmostEqual(result.ls_ret_net.iloc[0], 0.099)
        self.assertAlmostEqual(result.turnover.iloc[0], 0.5)

    def test_same_names_need_rebalancing_after_price_drift(self):
        result = self.e.compute_portfolio_returns(panel(((0.1, -0.1), (0., 0.))), transaction_cost=0.)
        self.assertAlmostEqual(result.turnover.iloc[1], (0.5 - 0.45 / 1.1) / 2)

    def test_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(self.e.compute_max_drawdown(pd.Series([-.2, .1])), .2)

    def test_baskets_cannot_overlap(self):
        with self.assertRaises(ValueError):
            self.e.compute_portfolio_returns(panel().iloc[:1])

    def test_long_only_has_no_short_leg(self):
        result = self.e.compute_portfolio_returns(panel(), short_pct=0, short_weight=0, long_weight=1, transaction_cost=0)
        self.assertEqual(result.n_short.iloc[0], 0)
        self.assertAlmostEqual(result.ls_ret.iloc[0], .1)

    def test_invalid_observations_are_rejected(self):
        for value in [np.nan, np.inf]:
            frame = panel()
            frame.loc[0, 'prediction'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.e.compute_portfolio_returns(frame)
        with self.assertRaises(ValueError):
            self.e.compute_portfolio_returns(pd.concat([panel(), panel()]))

    def test_no_fake_value_weighting_or_loss_floor(self):
        with self.assertRaises(ValueError):
            self.e.compute_portfolio_returns(panel(), weighting='value')
        with self.assertRaises(ValueError):
            self.e.compute_portfolio_returns(panel(((0., 3.),)))

    def test_monthly_portfolio_rejects_missing_months(self):
        frame = panel(((.1, -.1), (.1, -.1)))
        frame.loc[frame.date == frame.date.max(), 'date'] += pd.DateOffset(months=1)
        with self.assertRaises(ValueError):
            self.e.compute_portfolio_returns(frame)

