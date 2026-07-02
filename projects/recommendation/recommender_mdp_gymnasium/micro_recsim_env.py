import gymnasium as gym
import numpy as np


class MicroRecSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=10.0,
            shape=(6,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._true_preference = np.zeros(5, dtype=np.float32)
        self._fatigue = np.zeros(5, dtype=np.float32)
        self._patience = 10.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = self.np_random
        self._true_preference = self._rng.uniform(0.2, 0.9, size=5).astype(np.float32)
        self._fatigue = np.zeros(5, dtype=np.float32)
        self._patience = 10.0
        observation = np.concatenate(
            [self._fatigue, np.array([self._patience], dtype=np.float32)]
        ).astype(np.float32)
        info = {}
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        click_probability = float(
            np.clip(
                self._true_preference[action] * (1.0 - self._fatigue[action]),
                0.0,
                1.0,
            )
        )
        clicked = bool(self._rng.random() < click_probability)

        self._fatigue[action] = np.clip(self._fatigue[action] + 0.2, 0.0, 1.0)
        other_indices = np.arange(5) != action
        self._fatigue[other_indices] = np.clip(
            self._fatigue[other_indices] * 0.9,
            0.0,
            1.0,
        )

        if not clicked:
            self._patience = max(0.0, self._patience - 1.0)

        terminated = self._patience <= 0.0
        truncated = False

        if terminated:
            reward = -10.0
        elif clicked:
            reward = 1.0
        else:
            reward = 0.0

        observation = np.concatenate(
            [self._fatigue, np.array([self._patience], dtype=np.float32)]
        ).astype(np.float32)
        info = {
            "clicked": clicked,
            "click_probability": click_probability,
            "true_preference": self._true_preference.copy(),
        }
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = MicroRecSimEnv()
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    print("Initial observation:", obs)
    print("Hidden user preference:", env._true_preference)

    for i in range(100):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        print(
            f"Step {i:02d} | Action {random_action} | "
            f"Reward {reward:>5} | Clicked {info['clicked']} | "
            f"ClickProb {info['click_probability']:.3f} | "
            f"Patience {obs[5]:.1f} | Fatigue {obs[:5]}"
        )

        if terminated or truncated:
            print(f"User left the app. Total reward: {total_reward}")
            break
