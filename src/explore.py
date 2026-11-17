import gymnasium as gym
import famnit_gym
env=gym.make('famnit_gym/Astronaut-v1')#headless sim
"""
we can set render_mode=None, but the env constructor defaults it to None anyways, so we can omit it as well
"""
heights=[]
speeds=[]
h_min,h_max=114,686
h_bins=12
s_min,s_max=-600,600
s_bins=10
def to_bin(value,v_min,v_max,n_bins):
	value=max(v_min,min(value,v_max))#for avoiding out of range
	converter=(value-v_min)/(v_max-v_min)
	b=int(converter*n_bins)
	return min(b,n_bins-1)#stays with in the bins
	
def discretize(obs):
	height=obs[0]
	speed=obs[1]
	return (to_bin(height,h_min,h_max,h_bins),to_bin(speed,s_min,s_max,s_bins))
	
for ep in range(20):
	observation,info=env.reset()
	done=False
	steps=0#to track how many frames each game survived
	while not done:
		action=env.action_space.sample()#random action
		observation,reward,terminated,truncated, info=env.step(action)
		print(observation,"->",discretize(observation))
		heights.append(observation[0])
		speeds.append(observation[1])
		steps+=1
		done=terminated or truncated
	print(f"episode {ep}: {steps} steps, crashed={terminated},how={info['collision']}, last reward={reward}")



env.close()
		
		

