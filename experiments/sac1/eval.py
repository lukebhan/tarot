import gym
import gym_tarot
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env=gym.make('Tarot-v0')

#policy_kwargs = dict(net_arch=[256, 256])
policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
#model.learn(5000000)
model.load('tmp/best_model')
obs = env.reset()
term = False
xarr = []
yarr = []
xrefarr = []
yrefarr = []
total_rew = 0
while not term:
    action = model.predict(obs)
    action = np.zeros(4)
    obs, reward, term, prints = env.step(action)
    print(reward)
    xarr.append(prints["x"])
    xrefarr.append(prints["xref"])
    yarr.append(prints["y"])
    yrefarr.append(prints["yref"])
    total_rew += reward
np.savetxt('x', xarr)
np.savetxt('xref', xrefarr)
np.savetxt('y', yarr)
np.savetxt('yref', yrefarr)
print(total_rew)
