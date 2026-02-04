import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

def evaluate_v1():
    # Mêmes réglages que l'entraînement
    config_v1 = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
        },
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True
        },
        "lanes_count": 2,
        "duration": 60,
        "collision_reward": -5 # Important pour que la visualisation respecte la logique
    }

    env = gym.make("highway-v0", render_mode='human', config=config_v1)

    try:
        model = DQN.load("models/agent_v1_dqn", env=env)
        print("✅ Modèle V1 chargé.")
    except:
        print("❌ Modèle pas encore prêt.")
        return

    print("👀 Démo V1 : Regarde si elle double les voitures lentes !")
    
    for episode in range(5):
        obs, info = env.reset()
        done = truncated = False
        score = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            score += reward
            
        print(f"Épisode {episode + 1} : Score {score:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_v1()