"""Tests for ``TopKSelector`` enum + ``_get_topk_constrs`` strategies."""

__docformat__ = "restructuredtext"

import numpy as np

from wraact import TopKSelector
from wraact.oney._act import ActHullWithOneY


def _make_constrs(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Build ``(n, 1 + k + 1)`` constraint matrix with varied beta + coefs."""
    rng = np.random.default_rng(seed)
    const = rng.uniform(-1, 1, size=(n, 1))
    coefs = rng.uniform(-2, 2, size=(n, k))
    # Beta covers both signs and zero (filter must drop zero).
    beta = np.concatenate(
        [
            rng.uniform(-3, -0.5, size=(n // 2, 1)),
            rng.uniform(0.5, 3, size=(n - n // 2, 1)),
        ]
    )
    return np.hstack((const, coefs, beta))


class TestTopKSelectorEnum:
    """Enum surface."""

    def test_members(self):
        names = {m.name for m in TopKSelector}
        assert names == {
            "BETA_MIN",
            "BETA_ABS_MAX",
            "COEF_L1_MAX",
            "COEF_L1_MIN",
            "FIRST",
            "RANDOM",
        }

    def test_str_values(self):
        assert TopKSelector.BETA_MIN.value == "beta_min"
        assert TopKSelector.COEF_L1_MAX.value == "coef_l1_max"


class TestGetTopKConstrs:
    """``_get_topk_constrs`` selector dispatch."""

    def test_default_is_beta_min(self):
        """No selector arg = legacy BETA_MIN behavior."""
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(c, topk=3, is_min=True)
        expected = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.BETA_MIN
        )
        assert np.allclose(out, expected)

    def test_beta_min_picks_smallest_beta(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.BETA_MIN
        )
        # Output betas should all be <= every kept-but-not-picked beta.
        filtered = c[np.abs(c[:, -1]) > 1e-9]
        sorted_beta = np.sort(filtered[:, -1])
        assert set(out[:, -1].round(6)) == set(sorted_beta[:3].round(6))

    def test_beta_abs_max_picks_largest_abs(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.BETA_ABS_MAX
        )
        filtered = c[np.abs(c[:, -1]) > 1e-9]
        sorted_abs = np.sort(-np.abs(filtered[:, -1]))
        assert sorted(np.abs(out[:, -1]).round(6), reverse=True) == sorted(
            -sorted_abs[:3].round(6), reverse=True
        )

    def test_coef_l1_max(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.COEF_L1_MAX
        )
        # All 3 returned rows must be among the top-3 by L1 of input coefs.
        filtered = c[np.abs(c[:, -1]) > 1e-9]
        all_l1 = np.abs(filtered[:, 1:-1]).sum(axis=1)
        out_l1 = np.abs(out[:, 1:-1]).sum(axis=1)
        assert min(out_l1) >= np.sort(all_l1)[-3]

    def test_coef_l1_min(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.COEF_L1_MIN
        )
        filtered = c[np.abs(c[:, -1]) > 1e-9]
        all_l1 = np.abs(filtered[:, 1:-1]).sum(axis=1)
        out_l1 = np.abs(out[:, 1:-1]).sum(axis=1)
        assert max(out_l1) <= np.sort(all_l1)[2]

    def test_first_takes_n_after_filter(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=4, is_min=True, selector=TopKSelector.FIRST
        )
        filtered = c[np.abs(c[:, -1]) > 1e-9]
        assert out.shape[0] == min(4, filtered.shape[0])
        assert np.allclose(out, filtered[: out.shape[0]])

    def test_random_returns_topk(self):
        c = _make_constrs(10, 3)
        out = ActHullWithOneY._get_topk_constrs(
            c, topk=3, is_min=True, selector=TopKSelector.RANDOM
        )
        assert out.shape[0] == 3

    def test_empty_after_filter(self):
        c = np.zeros((5, 5))  # all beta = 0; filter drops everything.
        out = ActHullWithOneY._get_topk_constrs(c, topk=3)
        assert out.shape[0] == 0
