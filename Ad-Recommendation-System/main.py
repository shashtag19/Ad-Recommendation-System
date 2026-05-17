"""
Multi-Armed Bandit Ad Recommendation System
============================================
Entry point. Runs all three bandit algorithms against a simulated
ad environment and plots the results.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is on the path (fixes imports on Windows)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import AdEnvironment
from bandits import EpsilonGreedy, UCB, ThompsonSampling
from simulation import run_simulation, plot_results, print_report


def main():
    N_ADS    = 10
    N_ROUNDS = 5_000

    print("=" * 55)
    print("  Multi-Armed Bandit — Ad Recommendation System")
    print("=" * 55)

    env = AdEnvironment(n_ads=N_ADS, seed=42)
    env.summary()

    bandits = [
        EpsilonGreedy(N_ADS, epsilon=0.1),
        UCB(N_ADS, c=2.0),
        ThompsonSampling(N_ADS),
    ]

    all_rewards = []
    all_regrets = []
    all_counts  = []

    for bandit in bandits:
        print(f"  Running {bandit.name}...")
        rewards, regrets, counts = run_simulation(bandit, env, N_ROUNDS)
        all_rewards.append(rewards)
        all_regrets.append(regrets)
        all_counts.append(counts)

    print_report(env, bandits, all_rewards, all_regrets, all_counts, N_ROUNDS)
    plot_results(env, bandits, all_rewards, all_regrets, all_counts, N_ROUNDS)


if __name__ == "__main__":
    main()
