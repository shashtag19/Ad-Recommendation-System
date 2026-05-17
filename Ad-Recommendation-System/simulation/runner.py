import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def run_simulation(bandit, env, n_rounds: int):
    """
    Run a single bandit against the environment for n_rounds steps.

    Returns
    -------
    cumulative_rewards   : np.ndarray  — running total of clicks
    cumulative_regret    : np.ndarray  — running gap vs. always picking best ad
    ad_selection_counts  : np.ndarray  — how many times each ad was shown
    """
    cumulative_rewards  = np.zeros(n_rounds)
    cumulative_regret   = np.zeros(n_rounds)
    ad_selection_counts = np.zeros(env.n_ads, dtype=int)

    total_reward = 0
    total_regret = 0

    for t in range(n_rounds):
        ad     = bandit.select_ad()
        reward = env.show_ad(ad)
        bandit.update(ad, reward)

        total_reward += reward
        total_regret += env.best_ctr - env.true_ctrs[ad]
        ad_selection_counts[ad] += 1

        cumulative_rewards[t] = total_reward
        cumulative_regret[t]  = total_regret

    return cumulative_rewards, cumulative_regret, ad_selection_counts


def print_report(env, bandits, all_rewards, all_regrets, all_counts, n_rounds: int):
    """Print a summary table comparing all bandits after the simulation."""
    print("\n" + "═" * 55)
    print("  FINAL REPORT  —  after {:,} rounds".format(n_rounds))
    print("═" * 55)
    print(f"  Best ad (Ad {env.best_ad:02d}) true CTR : {env.best_ctr:.3f}")
    print()
    print(f"  {'Algorithm':<25} {'Clicks':>8} {'CTR':>8} {'Regret':>10}")
    print("  " + "─" * 51)

    for b, rewards, regrets, counts in zip(bandits, all_rewards, all_regrets, all_counts):
        clicks       = int(rewards[-1])
        achieved_ctr = clicks / n_rounds
        total_regret = regrets[-1]
        print(f"  {b.name:<25} {clicks:>8,} {achieved_ctr:>8.3f} {total_regret:>10.1f}")

    print()
    winner_idx = int(np.argmin([r[-1] for r in all_regrets]))
    print(f"  🏆  Lowest regret: {bandits[winner_idx].name}")
    print("═" * 55 + "\n")


def plot_results(env, bandits, all_rewards, all_regrets, all_counts, n_rounds: int,
                 save_path: str = "plots/bandit_results.png"):
    """
    Render a 4-panel figure:
      - Cumulative clicks over time
      - True ad CTRs (ground truth)
      - Cumulative regret over time
      - Ad selection heatmap
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor="#F7F3EF")
    fig.suptitle(
        "Multi-Armed Bandit — Ad Recommendation System",
        fontsize=18, fontweight="bold", color="#1A1A2E", y=0.97
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    rounds = np.arange(1, n_rounds + 1)

    # ── Cumulative Rewards ──────────────────────
    ax1.set_facecolor("#FAFAFA")
    for b, rewards in zip(bandits, all_rewards):
        ax1.plot(rounds, rewards, label=b.name, color=b.color, linewidth=2)
    ax1.set_title("Cumulative Clicks Over Time", fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Total Clicks")
    ax1.legend(framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    # ── True CTRs ───────────────────────────────
    ax2.set_facecolor("#FAFAFA")
    colors = ["#E94560" if i == env.best_ad else "#AAAAAA" for i in range(env.n_ads)]
    ax2.bar(range(env.n_ads), env.true_ctrs, color=colors, edgecolor="white")
    ax2.set_title("True Ad CTRs (hidden from bandits)", fontweight="bold")
    ax2.set_xlabel("Ad ID")
    ax2.set_ylabel("Click-Through Rate")
    ax2.set_xticks(range(env.n_ads))
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.annotate(
        "Best", xy=(env.best_ad, env.best_ctr),
        xytext=(env.best_ad + 0.5, env.best_ctr + 0.02),
        fontsize=8, color="#E94560", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E94560", lw=1.2)
    )

    # ── Cumulative Regret ────────────────────────
    ax3.set_facecolor("#FAFAFA")
    for b, regret in zip(bandits, all_regrets):
        ax3.plot(rounds, regret, label=b.name, color=b.color, linewidth=2)
    ax3.set_title("Cumulative Regret Over Time  (lower = better)", fontweight="bold")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Expected Regret")
    ax3.legend(framealpha=0.8)
    ax3.grid(True, alpha=0.3)

    # ── Ad Selection Heatmap ─────────────────────
    ax4.set_facecolor("#FAFAFA")
    matrix = np.array(all_counts)
    im = ax4.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax4.set_title("Ad Selection Counts", fontweight="bold")
    ax4.set_xlabel("Ad ID")
    ax4.set_ylabel("Algorithm")
    ax4.set_xticks(range(env.n_ads))
    ax4.set_yticks(range(len(bandits)))
    ax4.set_yticklabels([b.name.split(" ")[0] for b in bandits], fontsize=9)
    plt.colorbar(im, ax=ax4, shrink=0.8, label="# times shown")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"📈  Plot saved to {save_path}")
