# tests/test_agents.py

import unittest
from envs.carla_env import CarlaEnv
from rl_agents.v0_agent import AgentV0


class TestAgent(unittest.TestCase):
    def test_agent_training(self):
        try:
            env = CarlaEnv({"town": "Town01"})
        except Exception as e:
            self.skipTest("CARLA server non disponible : " + str(e))
            return
        agent = AgentV0(env, {"learning_rate": 0.0003})
        # Entraîner sur très peu de timesteps pour tester
        model = agent.train(total_timesteps=100)
        self.assertIsNotNone(model)
        # Vérifier que la méthode act retourne une action de bonne forme
        obs = env.reset()
        action = agent.act(obs)
        self.assertIsInstance(action, (list, tuple, np.ndarray))


if __name__ == "__main__":
    unittest.main()
