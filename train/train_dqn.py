from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import DQN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cyber_env import CyberSecurityEnv


def evaluate_action_distribution(model, env, sample_size=5000):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    rng = np.random.default_rng(7)

    for _ in range(sample_size):
        idx = int(rng.integers(0, len(env.data)))
        obs = env.data[idx]
        action, _ = model.predict(obs, deterministic=True)
        counts[int(action)] += 1

    total = max(sum(counts.values()), 1)
    print("\nAction distribution check:")
    for action in sorted(counts):
        percentage = (counts[action] / total) * 100
        print(f"Action {action}: {counts[action]} ({percentage:.2f}%)")

    dominant_ratio = max(counts.values()) / total
    if dominant_ratio > 0.95:
        print("Warning: Policy is still highly collapsed to one action.")
    else:
        print("Success: Policy is producing diverse actions.")

def main():
    env = CyberSecurityEnv(str(PROJECT_ROOT / "dataset" / "data.csv"))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=50000,
        batch_size=64,
        learning_starts=2000,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        target_update_interval=1000,
        verbose=1
    )

    model.learn(total_timesteps=250000)

    model.save(str(PROJECT_ROOT / "dqn_cyber_model"))
    evaluate_action_distribution(model, env, sample_size=5000)

if __name__ == "__main__":
    main()