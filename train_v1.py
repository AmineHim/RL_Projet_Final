import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

def train_v1():
    # --- CONFIGURATION V1 (Changement de voie + Punition Sévère) ---
    config_v1 = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True  # <--- CHANGEMENT MAJEUR : On autorise le changement de voie
        },
        "lanes_count": 2,
        "duration": 60,  # Épisodes plus longs
        "collision_reward": -5,  # <--- CORRECTION : Crash très punitif pour forcer la prudence
        "high_speed_reward": 0.4, # On garde la récompense de vitesse
        "lane_change_reward": 0,  # Pas de bonus gratuit pour changer de voie (il doit le faire utilement)
    }

    env = gym.make("highway-v0", render_mode=None, config=config_v1)

    # --- MODÈLE IA ---
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./logs/v1/",
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.4, # Un peu plus d'exploration car il y a plus d'actions possibles
    )

    print("🚗 Début de l'entraînement V1 (Changement de voie)...")
    print("La tâche est plus complexe, on entraîne sur 30 000 pas.")
    
    # On augmente le temps d'entraînement car il y a 2 dimensions (Vitesse + Direction)
    model.learn(total_timesteps=30000)

    path = "models/agent_v1_dqn"
    model.save(path)
    print(f"✅ Agent V1 sauvegardé sous : {path}.zip")

if __name__ == "__main__":
    train_v1()