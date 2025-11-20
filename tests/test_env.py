# tests/test_env.py

import unittest
from envs.carla_env import CarlaEnv


class TestCarlaEnv(unittest.TestCase):
    def test_reset_and_step(self):
        try:
            env = CarlaEnv({"town": "Town01"})
            obs = env.reset()
        except Exception as e:
            self.skipTest("CARLA server non disponible : " + str(e))
            return
        # Vérifier que l'observation contient les clés attendues
        self.assertIsInstance(obs, dict)
        self.assertIn("speed", obs)
        self.assertIn("lane_offset", obs)
        # Appliquer une action nominale (50% throttle, 0 steer)
        new_obs, reward, done, info = env.step([0.5, 0.0])
        self.assertIsInstance(new_obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
