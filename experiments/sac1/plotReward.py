import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
arr = []
count = 0
with open('tmp/monitor.csv') as f:
    line = f.readline()
    while line:
        if(count < 2):
            count += 1
            line = f.readline()
        else:
            val = line.split(",")
            print(val)
            line=f.readline()
            arr.append(float(val[0]))
plt.plot(arr)
plt.yscale('log')
plt.savefig("fig.png")
plt.show()
