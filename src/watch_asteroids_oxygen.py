import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.astronaut import Asteroids, Oxygen, DebugOverlay
import numpy as np

# Open a window this time (render_mode='human') and add the debug overlay,
# which shows the positions, the reward, the steps and the oxygen count.
env = gym.make('famnit_gym/Astronaut-v1', render_mode='human')
env = Asteroids(env)
env = Oxygen(env)
env = DebugOverlay(env)

# The trained table from train_asteroids_oxygen.py
Q = np.load('results/qtables/asteroids_oxygen_qtable.npy')

# The same constants as in training, so every state maps to the same bins.
astronaut_x = 300
h_min, h_max = 114, 686
h_bins = 12
s_min, s_max = -600, 600
s_bins = 10
dx_min, dx_max = 0, 1000
dx_bins = 6
dy_min, dy_max = -400, 400
dy_bins = 10
ox_min, ox_max = 0, 1000
ox_bins = 3
oy_min, oy_max = -400, 400
oy_bins = 6
frame_skip = 6

games = 5  # how many games to watch


def to_bin(value, v_min, v_max, n_bins):
	value = max(v_min, min(value, v_max))
	converter = (value - v_min) / (v_max - v_min)
	b = int(converter * n_bins)
	return min(b, n_bins - 1)


def discretize(obs):
	height = obs[0]
	speed = obs[1]
	dx = obs[2] - astronaut_x
	dy = obs[3] - height
	ox = obs[4] - astronaut_x
	oy = obs[5] - height
	return (to_bin(height, h_min, h_max, h_bins),
		to_bin(speed, s_min, s_max, s_bins),
		to_bin(dx, dx_min, dx_max, dx_bins),
		to_bin(dy, dy_min, dy_max, dy_bins),
		to_bin(ox, ox_min, ox_max, ox_bins),
		to_bin(oy, oy_min, oy_max, oy_bins))


for i in range(games):
	# the same seeds as the test games, so you watch games it never trained on
	obs, info = env.reset(seed=100000 + i)
	done = False
	while not done:
		action = int(np.argmax(Q[discretize(obs)]))  # always the best action, no exploring
		for _ in range(frame_skip):  # hold the action for 6 frames, like in training
			obs, r, terminated, truncated, info = env.step(action)
			if terminated or truncated:
				break
		done = terminated or truncated
	print(f"game {i}: {info['steps']} frames, {info['oxygen_count']} tanks, ended by {info['collision'] if terminated else 'time limit'}")

env.close()
