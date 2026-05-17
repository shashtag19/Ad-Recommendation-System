"""
Tests for AdEnvironment.
Run with: pytest tests/test_environment.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import AdEnvironment


def test_environment_creates_correct_number_of_ads():
    env = AdEnvironment(n_ads=8)
    assert len(env.true_ctrs) == 8

def test_environment_ctrs_in_valid_range():
    env = AdEnvironment(n_ads=20)
    assert np.all(env.true_ctrs >= 0.0)
    assert np.all(env.true_ctrs <= 1.0)

def test_environment_best_ad_is_correct():
    env = AdEnvironment(n_ads=10)
    assert env.best_ad == int(np.argmax(env.true_ctrs))
    assert env.best_ctr == env.true_ctrs[env.best_ad]

def test_environment_show_ad_returns_0_or_1():
    env = AdEnvironment(n_ads=5)
    for ad in range(env.n_ads):
        for _ in range(20):
            result = env.show_ad(ad)
            assert result in (0, 1)

def test_environment_seed_reproducibility():
    env1 = AdEnvironment(n_ads=10, seed=99)
    env2 = AdEnvironment(n_ads=10, seed=99)
    np.testing.assert_array_equal(env1.true_ctrs, env2.true_ctrs)

def test_environment_different_seeds_differ():
    env1 = AdEnvironment(n_ads=10, seed=1)
    env2 = AdEnvironment(n_ads=10, seed=2)
    assert not np.array_equal(env1.true_ctrs, env2.true_ctrs)
