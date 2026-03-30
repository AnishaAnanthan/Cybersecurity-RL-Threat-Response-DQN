from stable_baselines3 import DQN
from env.cyber_env import CyberSecurityEnv

def main():
    env = CyberSecurityEnv("dataset/data.csv")
    model = DQN.load("dqn_cyber_model")

    obs, _ = env.reset()

    for i in range(20):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        print(f"Step {i+1} → Action: {action}, Reward: {reward}")

        if done:
            break

if __name__ == "__main__":
    main()