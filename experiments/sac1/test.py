import gym
import gym_tarot
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env=gym.make('Tarot-v0')
env=Monitor(env, log_dir)

policy_kwargs = dict(net_arch=dict(pi=[256, 128, 128], qf=[256, 128, 128]))
model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(5000000)
model.save('model_no_fault')
