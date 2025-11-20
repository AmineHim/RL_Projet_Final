# rl_agents/v1_agent.py

from rl_agents.v0_agent import AgentV0


class AgentV1(AgentV0):
    """
    Agent RL pour V1 (dépassement par changement de voie).
    Hérite de AgentV0, on pourrait ici adapter l'architecture si besoin.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        # On utilise la même architecture PPO pour V1 (peut-être ré-entraîner sur l'env V1)

    # Pas de modification nécessaire ici; on peut réutiliser train() et act() de AgentV0.
    # On pourrait charger les poids de l'agent V0 puis continuer l'entraînement, etc.
