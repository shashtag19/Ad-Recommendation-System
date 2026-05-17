"""
Tests for bandit algorithms.
Run with: pytest tests/test_bandits.py -v
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bandits import EpsilonGreedy, UCB, ThompsonSampling

N_ADS = 5


# ── EpsilonGreedy ─────────────────────────────

def test_epsilon_greedy_select_returns_valid_index():
    bandit = EpsilonGreedy(N_ADS)
    for _ in range(50):
        assert 0 <= bandit.select_ad() < N_ADS

def test_epsilon_greedy_update_changes_values():
    bandit = EpsilonGreedy(N_ADS)
    bandit.update(0, 1)
    assert bandit.values[0] > 0
    assert bandit.counts[0] == 1

def test_epsilon_greedy_exploits_best_arm():
    bandit = EpsilonGreedy(N_ADS, epsilon=0.0)   # pure exploit
    bandit.values[3] = 0.9                        # arm 3 is best
    assert bandit.select_ad() == 3

def test_epsilon_greedy_estimated_ctrs_shape():
    bandit = EpsilonGreedy(N_ADS)
    assert bandit.estimated_ctrs().shape == (N_ADS,)


# ── UCB ───────────────────────────────────────

def test_ucb_select_returns_valid_index():
    bandit = UCB(N_ADS)
    for _ in range(50):
        idx = bandit.select_ad()
        assert 0 <= idx < N_ADS

def test_ucb_visits_all_arms_first():
    bandit = UCB(N_ADS)
    selected = set()
    for _ in range(N_ADS):
        selected.add(bandit.select_ad())
        bandit.update(len(selected) - 1, 0)
    assert selected == set(range(N_ADS))

def test_ucb_update_increments_count():
    bandit = UCB(N_ADS)
    bandit.update(2, 1)
    assert bandit.counts[2] == 1

def test_ucb_estimated_ctrs_shape():
    bandit = UCB(N_ADS)
    assert bandit.estimated_ctrs().shape == (N_ADS,)


# ── ThompsonSampling ──────────────────────────

def test_thompson_select_returns_valid_index():
    bandit = ThompsonSampling(N_ADS)
    for _ in range(50):
        assert 0 <= bandit.select_ad() < N_ADS

def test_thompson_update_increments_alpha_on_click():
    bandit = ThompsonSampling(N_ADS)
    before = bandit.alpha[1]
    bandit.update(1, 1)
    assert bandit.alpha[1] == before + 1
    assert bandit.beta[1] == 1   # unchanged

def test_thompson_update_increments_beta_on_no_click():
    bandit = ThompsonSampling(N_ADS)
    before = bandit.beta[1]
    bandit.update(1, 0)
    assert bandit.beta[1] == before + 1
    assert bandit.alpha[1] == 1  # unchanged

def test_thompson_estimated_ctrs_between_0_and_1():
    bandit = ThompsonSampling(N_ADS)
    ctrs = bandit.estimated_ctrs()
    assert np.all(ctrs >= 0) and np.all(ctrs <= 1)

def test_thompson_estimated_ctrs_shape():
    bandit = ThompsonSampling(N_ADS)
    assert bandit.estimated_ctrs().shape == (N_ADS,)
