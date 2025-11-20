# rl_agents/v0_agent.py

from stable_baselines3 import PPO


class AgentV0:
    """
    Agent RL pour V0 (conduite simple) utilisant PPO.
    """

    def __init__(self, env, config):
        """
        env : instance de CarlaEnv
        config : dictionnaire de paramètres d'entraînement (learning_rate, gamma, etc.)
        """
        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.get("learning_rate", 3e-4),
            gamma=config.get("gamma", 0.99),
            verbose=1,
        )

    def train(self, total_timesteps):
        """
        Entraîne le modèle pendant total_timesteps pas de simulation.
        """
        self.model.learn(total_timesteps=total_timesteps)
        return self.model

    def act(self, observation):
        """
        Retourne l'action prédite (sans exploration) pour une observation donnée.
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return action
