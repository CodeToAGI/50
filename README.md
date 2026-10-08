# EP50 — Reinforcement Learning Explained  
### Agents, Environments, Rewards & Tabular Q-Learning on CartPole

**Deep Learning Series · Module 10 · Episode 50 of 72**

An agent acts. The world replies with a number. There is no answer key.

This episode builds a complete tabular Q-learning agent from scratch in pure NumPy that learns to balance the classic CartPole.

---

## What you will learn

- The full RL loop (agent ↔ environment)
- Markov Decision Process + discount factor γ
- Exploration vs Exploitation (ε-greedy + live bandit experiment)
- Q-learning: the Q-table and the Bellman update
- Discretising continuous state into a table
- Training schedules for ε and α
- Honest multi-seed evaluation (not just the lucky run)

---

## Quick Start

```bash
pip install numpy gymnasium matplotlib
python ep50_qlearning.py
