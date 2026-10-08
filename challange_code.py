"""EP50 - Tabular Q-learning on CartPole: an RL agent that learns from trial and reward.
    python ep50_qlearning.py        # train 5000 episodes, evaluate, save a learning curve
pip install numpy gymnasium matplotlib
"""
import numpy as np
import gymnasium as gym

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEED = 0
EPISODES = 5000
BINS = (1, 1, 6, 12)                   # cart position, cart speed, pole angle, pole spin
LOW = (-2.4, -3.0, -0.21, -3.5)
HIGH = (2.4, 3.0, 0.21, 3.5)
ALPHA0, ALPHA_MIN = 0.5, 0.05          # learning rate, decays linearly
GAMMA = 0.99                           # how much future reward counts
EPS0, EPS_MIN = 1.0, 0.01              # exploration: 100% random -> 1% random
EXPLORE_FRAC = 0.8                     # epsilon hits its floor at ~80% of training


# ── 2. STATE: chop 4 continuous numbers into ONE cell of a table ──────────
EDGES = [np.linspace(lo, hi, b - 1) for lo, hi, b in zip(LOW, HIGH, BINS)]


def discretise(obs):
    return tuple(int(np.digitize(x, e)) for x, e in zip(obs, EDGES))


# ── 3. AGENT: a Q-table, epsilon-greedy choice, the Bellman update ────────
Q = np.zeros(BINS + (2,))              # Q[state][action] = expected future reward


def choose(state, eps, rng):
    if rng.random() < eps:             # explore: try something random
        return int(rng.integers(2))
    return int(Q[state].argmax())      # exploit: the best action found so far


def learn(s, a, r, s2, terminated, alpha):
    target = r + (0.0 if terminated else GAMMA * Q[s2].max())
    Q[s + (a,)] += alpha * (target - Q[s + (a,)])


# ── 4. TRAIN: play, get rewards, update the table ─────────────────────────
def train(env, rng):
    returns = []
    for ep in range(EPISODES):
        frac = ep / EPISODES
        eps = max(EPS_MIN, EPS0 * (1 - frac / EXPLORE_FRAC))
        alpha = max(ALPHA_MIN, ALPHA0 * (1 - frac))
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        s, done, total = discretise(obs), False, 0.0
        while not done:
            a = choose(s, eps, rng)
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = discretise(obs)
            learn(s, a, r, s2, terminated, alpha)
            s, total, done = s2, total + r, terminated or truncated
        returns.append(total)
        if (ep + 1) % 500 == 0:
            print(f"episode {ep + 1:5d}  avg return (last 100): "
                  f"{np.mean(returns[-100:]):6.1f}  epsilon {eps:.2f}")
    return returns


# ── 5. EVALUATE: greedy play, no exploration, then save the curve ─────────
def evaluate(env, n=100):
    scores = []
    for k in range(n):
        obs, _ = env.reset(seed=10_000 + k)
        s, done, total = discretise(obs), False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(int(Q[s].argmax()))
            s, total, done = discretise(obs), total + r, terminated or truncated
        scores.append(total)
    return float(np.mean(scores))


if __name__ == "__main__":
    env, rng = gym.make("CartPole-v1"), np.random.default_rng(SEED)
    curve = train(env, rng)
    print(f"greedy evaluation, 100 episodes: {evaluate(env):.1f} / 500")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        avg = np.convolve(curve, np.ones(100) / 100, mode="valid")
        plt.plot(avg)
        plt.xlabel("episode")
        plt.ylabel("average return (100 episodes)")
        plt.savefig("ep50_learning_curve.png", dpi=120)
        print("saved ep50_learning_curve.png")
    except ImportError:
        print("matplotlib not installed - skipping the plot")
