import gymnasium as gym
import famnit_gym
import numpy as np
import time

env=gym.make('famnit_gym/Astronaut-v1')



"""constants
"""
seed=0

h_min,h_max=114,686
h_bins=12
s_min,s_max=-600,600
s_bins=10
alpha=0.1
gamma=0.99
Q=np.zeros((h_bins,s_bins,2))#we have height_bins*speed_bins*2(for thrust ON/OFF) combinations
frame_skip=6
episodes=3000
epsilon=0.1

np.random.seed(seed)
env.action_space.seed(seed)

def to_bin(value,v_min,v_max,n_bins):
	value=max(v_min,min(value,v_max))#for avoiding out of range
	converter=(value-v_min)/(v_max-v_min)
	b=int(converter*n_bins)
	return min(b,n_bins-1)#stays with in the bins
	
def discretize(obs):
	height=obs[0]
	speed=obs[1]
	return (to_bin(height,h_min,h_max,h_bins),to_bin(speed,s_min,s_max,s_bins))

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
	results=[]
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
		results.append(info['steps'])
	return results

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
	if ep%500==0:
		print(ep,info['steps'],round(log[-1][1],5))
		
train_time=time.time()-start
trained=evaluate(True)
rand=evaluate(False)
print("trained agent: mean",np.mean(trained),"frames,",sum(s==1200 for s in trained),"/50 survivced")
print("random agent: mean",np.mean(rand),"frames,", sum(s==1200 for s in rand),"/50 survived")
print("training time: ",round(train_time,1),"s")
np.savetxt('results/logs/no_wrapper_log.csv',log,delimiter=',',header='episode,loss,steps',comments='',fmt=['%d','%.8f','%d'])
np.save('results/qtables/no_wrapper_qtable.npy',Q)
print("saved results/logs/no_wrapper_log.csv and results/qtables/no_wrapper_qtable.npy")
		



		
