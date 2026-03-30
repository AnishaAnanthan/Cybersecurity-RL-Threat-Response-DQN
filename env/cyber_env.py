import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CyberSecurityEnv(gym.Env):
    def __init__(self, data_path):
        super(CyberSecurityEnv, self).__init__()

        df = pd.read_csv(data_path)

        # Keep numeric only
        df = df.select_dtypes(include=[np.number]).fillna(0)

        # Take first 10 features
        df = df.iloc[:, :10]

        # Normalize safely
        min_vals = df.min()
        max_vals = df.max()
        df = (df - min_vals) / (max_vals - min_vals + 1e-8)
        df = df.clip(0, 1)

        self.data = df.values.astype(np.float32)
        self.feature_names = df.columns.tolist()
        self.current_step = 0
        self.episode_step = 0
        self.max_episode_steps = 200

        # Use intensity quartiles so each severity band is represented in training.
        intensities = self.data.mean(axis=1)
        self.intensity_q1, self.intensity_q2, self.intensity_q3 = np.quantile(
            intensities, [0.25, 0.50, 0.75]
        )

        # Action risk order: allow(0) < alert(3) < monitor(2) < block(1)
        self.action_risk = {0: 0, 3: 1, 2: 2, 1: 3}

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(len(self.data) - self.max_episode_steps - 1, 0)
        self.current_step = int(self.np_random.integers(0, max_start + 1))
        self.episode_step = 0

        return self.data[self.current_step], {}

    def step(self, action):
        state = self.data[self.current_step]

        traffic_intensity = float(np.mean(state))

        if traffic_intensity < self.intensity_q1:
            severity_band = 0
        elif traffic_intensity < self.intensity_q2:
            severity_band = 1
        elif traffic_intensity < self.intensity_q3:
            severity_band = 2
        else:
            severity_band = 3

        action_risk = self.action_risk[int(action)]
        distance = abs(action_risk - severity_band)

        # Highest reward when action risk matches severity; lower as mismatch grows.
        reward = 30 - 15 * distance

        # Extra penalties for clearly wrong extremes.
        if severity_band == 3 and int(action) == 0:
            reward -= 20
        if severity_band == 0 and int(action) == 1:
            reward -= 10

        # Move forward
        self.current_step += 1
        self.episode_step += 1
        done = (
            self.current_step >= len(self.data) - 1
            or self.episode_step >= self.max_episode_steps
        )

        next_state = self.data[self.current_step]

        return next_state, reward, done, False, {}

    def render(self):
        pass