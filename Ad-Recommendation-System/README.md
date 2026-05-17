# Multi-Armed Bandit — Ad Recommendation System

A Python simulation that pits three bandit algorithms against each other to find the best-performing ad, without knowing click-through rates in advance.

---

## How It Works

Each ad is an "arm". The bandit shows an ad, observes whether the user clicks, and updates its belief about that ad's value. The challenge is the **explore/exploit tradeoff** — do you keep showing the best ad you've found so far, or try others that might be even better?

### Algorithms Compared

| Algorithm | Strategy |
|---|---|
| **Epsilon-Greedy** | Explores randomly 10% of the time, exploits the best known ad otherwise |
| **UCB** | Adds a confidence bonus to uncertain arms — naturally explores more when unsure |
| **Thompson Sampling** | Maintains a probability distribution per arm and samples from it (Bayesian) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
ad-bandit/
├── main.py                        # Entry point
├── requirements.txt
├── .gitignore
│
├── bandits/                       # One file per algorithm
│   ├── __init__.py
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   └── thompson_sampling.py
│
├── environment/                   # Ad server simulation
│   ├── __init__.py
│   └── ad_environment.py
│
├── simulation/                    # Runner, plotting, reporting
│   ├── __init__.py
│   └── runner.py
│
├── plots/                         # Output charts saved here
│   └── .gitkeep
│
└── tests/
    ├── test_bandits.py
    └── test_environment.py
```

---

## Output

Running `main.py` prints a report like this:

```
═══════════════════════════════════════════════════════
  FINAL REPORT  —  after 5,000 rounds
═══════════════════════════════════════════════════════
  Best ad (Ad 07) true CTR : 0.334

  Algorithm                   Clicks      CTR     Regret
  ───────────────────────────────────────────────────────
  Epsilon-Greedy (ε=0.1)       1,432    0.286      142.3
  UCB                          1,489    0.298       98.7
  Thompson Sampling            1,501    0.300       81.2

  🏆  Lowest regret: Thompson Sampling
```

And saves a 4-panel chart to `plots/bandit_results.png`.

---

## Possible Extensions

- **Contextual bandits** — use user features (age, location, device) to personalise ad selection
- **Decaying epsilon** — reduce exploration rate as the model gains confidence
- **Non-stationary rewards** — CTRs drift over time, forcing the bandit to re-explore
- **Budget constraints** — each ad has a cost; maximise clicks per dollar spent
- **LinUCB** — a linear contextual bandit, standard in real ad systems

---

## Concepts

- [Multi-Armed Bandit Problem — Wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Thompson Sampling — Russo et al. 2018](https://arxiv.org/abs/1707.02038)
- [UCB1 — Auer et al. 2002](https://link.springer.com/article/10.1023/A:1013689704352)
