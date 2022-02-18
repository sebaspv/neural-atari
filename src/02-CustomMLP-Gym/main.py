import gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

policy_kwargs = {
    "activation_fn": torch.nn.ReLU,
    "net_arch": [{
        "pi": [128, 64, 32],
        "vf": [128, 64, 32]
        }],
}

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

model.save('cartpole_custompolicy')
