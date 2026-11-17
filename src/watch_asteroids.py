import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.astronaut import Asteroids, DebugOverlay
import numpy as np

# same settings as train_asteroids.py
astronaut_x=300
h_min,h_max=114,686
h_bins=12
s_min,s_max=-600,600
s_bins=10
dx_min,dx_max=0,1000
dx_bins=6
dy_min,dy_max=-400,400
dy_bins=10
frame_skip=6

Q=np.load('asteroids_qtable.npy')		# the trained table

env=gym.make('famnit_gym/Astronaut-v1',render_mode='human')	# with a window
env=Asteroids(env)
env=DebugOverlay(env)		# draws collision circles, positions, step count

def to_bin(value,v_min,v_max,n_bins):
	value=max(v_min,min(value,v_max))
	converter=(value-v_min)/(v_max-v_min)
	b=int(converter*n_bins)
	return min(b,n_bins-1)

def discretize(obs):
	height=obs[0]
	speed=obs[1]
	dx=obs[2]-astronaut_x
	dy=obs[3]-height
	return (to_bin(height,h_min,h_max,h_bins),to_bin(speed,s_min,s_max,s_bins),to_bin(dx,dx_min,dx_max,dx_bins),to_bin(dy,dy_min,dy_max,dy_bins))

def do_action(action):
	total_rewards=0
	for _ in range(frame_skip):
		obs,r,terminated,truncated,info=env.step(action)
		total_rewards+=r
		if terminated or truncated:
			break
	return obs,total_rewards,terminated,truncated,info

for game in range(3):
	obs,info=env.reset(seed=100000+game)		# test games it never trained on
	done=False
	while not done:
		action=int(np.argmax(Q[discretize(obs)]))	# epsilon = 0: best action only
		obs,r,terminated,truncated,info=do_action(action)
		done=terminated or truncated
	print("game",game,":",info['steps'],"frames, collision:",info['collision'])

env.close()
