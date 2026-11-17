import numpy as np
import matplotlib.pyplot as plt


name = 'asteroids'              # 'no_wrapper', 'asteroids', 'asteroids_oxygen'
title = 'Part 2 (asteroids)'
block_size = 200                      


data = np.loadtxt('results/logs/' + name + '_log.csv', delimiter=',', skiprows=1)
losses = data[:, 1]              #the loss of every game
frames = data[:, 2]              #the frames survived in every game


def average_per_block(values):
    block_starts = []
    block_averages = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        block_starts.append(start)
        block_averages.append(np.mean(block))
    return block_starts, block_averages

loss_x, loss_y = average_per_block(losses)
frames_x, frames_y = average_per_block(frames)


fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

top.plot(loss_x, loss_y, marker='o')
top.set_title(title + ': loss convergence')
top.set_ylabel('average loss')

bottom.plot(frames_x, frames_y, marker='o')
bottom.set_title(title + ': frames survived')
bottom.set_xlabel('game')
bottom.set_ylabel('average frames survived (max 1200)')

plt.tight_layout()
plt.savefig('results/plots/' + name + '_plot.png', dpi=150)
print('saved', 'results/plots/' + name + '_plot.png')
