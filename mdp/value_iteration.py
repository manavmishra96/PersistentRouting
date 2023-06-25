import numpy as np
from tqdm import tqdm
import itertools


def transitions(state, action, next_state):
    global states, loc_space, clock_space, k_space
    indexer = np.arange(len(states)).reshape(len(loc_space), clock_space.shape[0], clock_space.shape[1], len(k_space))
    
    curr_loc, curr_clock_i, curr_clock_j, curr_k = np.transpose(np.where(indexer == state))[0]
    next_loc, next_clock_i, next_clock_j, next_k = np.transpose(np.where(indexer == next_state))[0]
    
    prob = 0.0
    
    if action == 0 and curr_k == K and next_loc == 0: # Condition to check if k == K then action is to return to depot
        prob = 1.0
    elif (curr_loc != next_loc) and (next_k == curr_k + 1) and (action == next_loc): # Condition to check for possible transition
        prob = 1.0 
            
    return prob


def rewards(state, action, next_state):
    global states, loc_space, clock_space, k_space
    indexer = np.arange(len(states)).reshape(len(loc_space), clock_space.shape[0], clock_space.shape[1], len(k_space))
    
    curr_loc, curr_clock_i, curr_clock_j, curr_k = np.transpose(np.where(indexer == state))[0]
    next_loc, next_clock_i, next_clock_j, next_k = np.transpose(np.where(indexer == next_state))[0]
    
    reward = -np.max(clock_space[curr_clock_i])
    return reward


def value_iteration(states, actions, discount_factor = 0.95, epsilon=1e-6, max_iterations=1000):
    num_states = len(states)
    num_actions = len(actions)
    
    V = np.zeros(num_states)  # Initialize state value function
    
    print("Starting Value Iteration...")
    for iteration in tqdm(range(max_iterations)):
        delta = 0
        for state in states:
            v = V[state]
            q_values = np.zeros(num_actions)

            # Calculate Q-value for each action
            for action in actions:
                for next_state in states:
                    transition_prob = transitions(state, action, next_state)
                    immediate_reward = rewards(state, action, next_state)
                    q_values[action] += transition_prob * (immediate_reward + discount_factor * V[next_state])

            V[state] = np.max(q_values)  # Update state value function
            delta = max(delta, abs(v - V[state]))

        if delta < epsilon:
            break

    print("Policy Rollout...")
    policy = np.zeros(num_states, dtype=int)
    for state in states:
        q_values = np.zeros(num_actions)

        # Calculate Q-value for each action
        for action in actions:
            for next_state in states:
                transition_prob = transitions[state, action, next_state]
                immediate_reward = rewards[state, action, next_state]
                q_values[action] += transition_prob * (immediate_reward + discount_factor * V[next_state])

        policy[state] = np.argmax(q_values)  # Select the action with the maximum Q-value

    return V, policy


if __name__ == "__main__":
    n = 3
    K = 7
    T = 7
    
    loc_space = np.arange(n + 1)
    k_space = np.arange(K + 1)
    clock_space = np.array(list(itertools.product(np.arange(T+1), repeat=n)))
    
    states = np.arange((len(loc_space) * clock_space.shape[0] * clock_space.shape[1] * len(k_space)))
    actions = np.arange(n + 1)
    
    value, policy = value_iteration(states, actions)
    print(policy)
     