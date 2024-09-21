import gym
import numpy as np
import random
env = gym.make('MountainCar-v0', render_mode = 'rgb_array')

env.reset()
ALPHA = 0.1 # Learning rate
DISCOUNT = 0.95 # Discount factor
EPISODES = 6000
SHOW_EVERY = 2000
# GAMMA = 0.99 # Reward discount
# EPSILON = 0.1
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

discrete_state = get_discrete_state(env.reset()[0])
done = False

discrete_state
for episode in range( EPISODES):
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset()[0])
    done = False
while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, _, _ = env.step(action)
    # print(new_state)
    new_discrete_state = get_discrete_state(new_state)
    # print(new_discrete_state)
    env.render()
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state + (action, )] = new_q
    elif new_state[0] >= env.goal_position:
        q_table[discrete_state + (action, )] = 0
    
    discrete_state = new_discrete_state
env.close()
