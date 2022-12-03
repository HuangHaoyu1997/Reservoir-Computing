import numpy as np
import matplotlib.pyplot as plt

with open('test1.log', 'r') as f:
    log = f.readlines()

acc, loss = [], []
for i in log:
    acc.append(float(i.split(',')[0][-6:]))
    loss.append(float(i.split(',')[1].split('\') ')[1].split('\n')[0]))
plt.subplot(121)
plt.plot(acc)
plt.axis([-5, 105, -0.05, 1.05])
plt.grid()

plt.subplot(122)
plt.plot(loss)
plt.grid()
plt.show()
