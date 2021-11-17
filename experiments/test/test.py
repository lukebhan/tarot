import gym
import gym_tarot
import numpy as np

env=gym.make('Tarot-v0')

result = False
obs = env.reset()

xarr = []
yarr = []
xrefarr = []
yrefarr = []
while not result:
    action = np.zeros(8)
    obs, reward, result, prints = env.step(action)
    xarr.append(prints["x"])
    xrefarr.append(prints["xref"])
    yarr.append(prints["y"])
    yrefarr.append(prints["yref"])
np.savetxt('x', xarr)
np.savetxt('xref', xrefarr)
np.savetxt('y', yarr)
np.savetxt('yref', yrefarr)

