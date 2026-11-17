import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.astronaut import Asteroids,Oxygen
import numpy as np
import time

seed=0
def temp_reward(info):
	if info['collision']:
		return -1
	if info['oxygen_picked']:
		return 1
	return 0
env=gym.make('famnit_gym/Astronaut-v1',options={'reward_function':temp_reward})
env=Asteroids(env)
env=Oxygen(env)
np.random.seed(seed)
env.action_space.seed(seed)


"""constants
"""
astronaut_x=300
ox_min,ox_max=0,1000
ox_bins=3
oy_min,oy_max=-400,400
oy_bins=6
dx_min,dx_max=0,1000
dx_bins=6
dy_min,dy_max=-400,400
dy_bins=10
episodes=60000


h_min,h_max=114,686
h_bins=12
s_min,s_max=-600,600
s_bins=10
alpha=0.1
gamma=0.99
Q=np.zeros((h_bins,s_bins,dx_bins,dy_bins,ox_bins,oy_bins,2))
"""
updated state,
"""
frame_skip=6

epsilon=0.1



def to_bin(value,v_min,v_max,n_bins):
	value=max(v_min,min(value,v_max))#for avoiding out of range
	converter=(value-v_min)/(v_max-v_min)
	b=int(converter*n_bins)
	return min(b,n_bins-1)#stays with in the bins
	
def discretize(obs):
	height=obs[0]
	speed=obs[1]
	dx=obs[2]-astronaut_x#using relative distance
	dy=obs[3]-height
	ox=obs[4]-astronaut_x
	oy=obs[5]-height
	
	
	return (to_bin(height,h_min,h_max,h_bins),to_bin(speed,s_min,s_max,s_bins),to_bin(dx,dx_min,dx_max,dx_bins),to_bin(dy,dy_min,dy_max,dy_bins),to_bin(ox,ox_min,ox_max,ox_bins),to_bin(oy,oy_min,oy_max,oy_bins))

def choose_action(state,epsilon):
	explore=np.random.choice([True,False],p=[epsilon,1-epsilon])
	if explore:
		return env.action_space.sample()#if explore==True we choose random
	return int(np.argmax(Q[state]))#otherwise we exploit(take the best value (greedy))

def do_action(action):
	total_rewards=0
	for _ in range(frame_skip):
		obs,r,terminated,truncated,info=env.step(action)
		total_rewards+=r
		if terminated or truncated:
			break
	return obs,total_rewards,terminated,truncated,info
	
log=[]#one entry per game
start=time.time()# clock before training

def evaluate(greedy,n=50):
	frames=[]
	tanks=[]
	for i in range(n):
		obs,info=env.reset(seed=100000+i)
		done=False
		while not done:
			if greedy:
				action=int(np.argmax(Q[discretize(obs)]))
			else:
				action=env.action_space.sample()
			obs,r,terminated,truncated,info=do_action(action)
			done=terminated or truncated
		frames.append(info['steps'])
		tanks.append(info['oxygen_count'])
	return frames,tanks

for ep in range(episodes):#outer loop just restarts the game when it ends
	obs,info=env.reset(seed=seed+ep)
	state=discretize(obs)
	done=False
	squared_errors=[]
	
	while not done:
		action=choose_action(state,epsilon)
		"""
		since epsilon=0.1, we 90% of the time we pick the bigger number, 105 of the time we pick randomly.
		then we do the picked action for 6 frames using 'do_actions()', then next state and we do it again
		"""
		obs,r,terminated,truncated,info=do_action(action)
		next_state=discretize(obs)
		if terminated:
			target=r
		else:
			target=r+gamma*np.max(Q[next_state])
		error=target-Q[state][action]
		Q[state][action]+=alpha*error
		squared_errors.append(error**2)
		state=next_state
		done=terminated or truncated
	log.append([ep,np.mean(squared_errors),info['steps']])
	if ep%2000==0:
		print(ep,info['steps'],round(log[-1][1],5))
		
train_time=time.time()-start
trained_frames,trained_tanks=evaluate(True)
random_frames,random_tanks=evaluate(False)
print("trained agent: mean",np.mean(trained_frames),"frames,",sum(s==1200 for s in trained_frames),"/50 survivced, mean tanks", np.mean(trained_tanks))
print("random agent: mean",np.mean(random_frames),"frames,", sum(s==1200 for s in random_frames),"/50 survived, mean tanks", np.mean(random_tanks))
print("training time: ",round(train_time,1),"s")
np.savetxt('asteroids_oxygen_with_temp_60k_log.csv',log,delimiter=',',header='episode,loss,steps',comments='',fmt=['%d','%.8f','%d'])
np.save('asteroids_oxygen_qtable_with_60k.npy',Q)
print("asteroids_oxygen_60k_log.csv and asteroids_oxygen_qtable_60k.npy")
		



		
