import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import highway_env


class SimpleDrivingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.env = gym.make("highway-v0", render_mode=render_mode)
        self.env.unwrapped.configure({
            "lanes_count": 1,
            "vehicles_count": 8,
            "action": {"type": "ContinuousAction", "steering_range": [0, 0], "lateral": False},
            "observation": {"type": "Kinematics", "vehicles_count": 5, "features": ["presence", "x", "y", "vx", "vy"]},
            "duration": 60,
        })
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = spaces.Box(-1, 1, (26,), np.float32)
        self.target_speed = 30.0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return self._obs(obs), info

    def _obs(self, obs):
        speed = np.clip(self.env.unwrapped.vehicle.speed / 45, 0, 1)
        return np.concatenate([[speed], obs.flatten()]).astype(np.float32)

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(np.array([action[0], 0], dtype=np.float32))
        return self._obs(obs), self._reward(), term, trunc, info

    def _reward(self):
        v = self.env.unwrapped.vehicle
        if v.crashed:
            return -15.0
        
        reward = v.speed / self.target_speed  # Vitesse
        dist = self._front_distance()
        
        if dist:
            if dist < 2: reward -= 5
            elif dist < 5: reward -= 2.3
            elif dist < 12: reward += (dist - 2) 
            elif dist < 20: reward += (dist - 7) 
            elif dist >= 20: reward -= 2.5
        
        return reward

    def _front_distance(self):
        v = self.env.unwrapped.vehicle
        dists = [other.position[0] - v.position[0] 
                 for other in self.env.unwrapped.road.vehicles 
                 if other is not v and 0 < other.position[0] - v.position[0] < 100 
                 and abs(other.position[1] - v.position[1]) < 2.5]
        return min(dists) if dists else None

    def render(self): return self.env.render()
    def close(self): self.env.close()


def train(timesteps=20000):
    env = VecNormalize(DummyVecEnv([lambda: SimpleDrivingEnv()] * 4), norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=128, verbose=1)
    model.learn(timesteps)
    model.save("driver")
    env.save("driver_norm.pkl")
    env.close()
    return model


def test(episodes=5):
    model = PPO.load("driver")
    env = VecNormalize.load("driver_norm.pkl", DummyVecEnv([lambda: SimpleDrivingEnv("human")]))
    env.training, env.norm_reward = False, False
    
    for ep in range(episodes):
        obs, total = env.reset(), 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward[0]
            if done: break
        print(f"Ep {ep+1}: reward={total:.1f}")
    env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        train()
        test()
