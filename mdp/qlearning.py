import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer  

# fix the number of targets 'n' and the maximum number of visits 'K' in each walk
n = 5
K = 7

T = 10

def combinations(lst_len, upr_lmt):
    '''
    takes the length of the list of targets and depot, and the upper limit of the clock and returns all possible clock combinations

    Parameters
    ----------
    lst_len : int
        length of the list
    upr_lmt : int
        upper limit of the clock

    Returns
    -------
    new_lst : list
        list of all possible clock combinations

    '''
    if lst_len == 2:
        new_lst = []
        for i in range(upr_lmt + 1):
            for j in range(upr_lmt + 1):
                new_lst.append([i, j])

        return new_lst

    else:
        sub_lst = combinations(lst_len - 1, upr_lmt)
        final_lst = []
        for i in range(upr_lmt + 1):
            for j in sub_lst:
                final_lst.append([i] + j)
        return final_lst

def transition_fxn(location, clocks, k_visited, action, K = K):
    '''
    generates the new location, updated clock readings and number of targets visited based on the current location, clocks readings,
    number of targets visited and the action

    Parameters
    ----------
    location : int
        current location
    clocks : numpy array
        current clock readings
    k_visited : int
        number of targets visited
    action : int
        action taken

    Returns
    -------
    location : int
        new location
    clocks : numpy array
        updated clock readings
    k_visited : int
        updated number of targets visited
    '''
    # increment each clock by 1
    clocks += 1

    # if any clock reading exceeds the upper limit, clip it to the upper limit
    clocks = np.clip(clocks, a_min = 0, a_max = K)

    # when the number of visits has become equal to the maximum number of visits, the agent can only go to the depot
    if k_visited == K:
        k_visited = 0
        location = 0
    
    else:
        k_visited += 1
        location = action
        
    clocks[location] = 0 # reset the clock reading of the new location to 0
    
    return location, clocks, k_visited

def reward_fxn(location, clocks, k_visited, action, K = K):
    '''
    returns the reward based on the current location, clocks readings, number of targets visited and the action

    Parameters
    ----------
    location : int
        current location
    clocks : numpy array  
        current clock readings
    k_visited : int
        number of targets visited
    action : int
        action taken

    Returns
    -------
    reward : int
        reward
    '''

    # the agent should not choose the action that takes it to the same location where it already is
    if location == action:
        return np.inf
    
    # when the number of visits has become equal to the maximum number of visits, the agent can only go to the depot
    if k_visited == K:
        if action == 0:
            return 0
        else:
            return np.inf
        
    else:
        clocks += 1
        return max(clocks[1:]) # the reward is the maximum of the clock readings of all targets (except the depot)
    

def qlearning(qtable, total_episodes, learning_rate, K, gamma):
    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.005            # Exponential decay rate for exploration prob
    
   # List of rewards
    rewards = []
    
    action_space = np.arange(n + 1)

    # 2 For life or until learning is stopped
    for episode in range(total_episodes):
        # Reset the environment
        state = 0
        step = 0
        done = False
        total_rewards = 0
        
        for step in range(K):
            # 3. Choose an action a in the current world state (s)
            ## First we randomize a number
            exp_exp_tradeoff = np.random.uniform(0, 1)
            
            ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])
                #print(exp_exp_tradeoff, "action", action)

            # Else doing a random choice --> exploration
            else:
                action = np.random.choice(action_space[1:])
                #print("action random", action)
                
            
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            
            total_rewards += reward
            
            # Our new state is state
            state = new_state
            
            # If done (if we're dead) : finish episode
            if done == True: 
                break
            
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)
    
    return qtable, rewards
    

def walk(loc, clk, k, opt_policy, K = K):
    '''
    simulates the agent's walk for 20 steps

    Parameters
    ----------
    loc0 : int
        initial location
    clk0 : numpy array
        initial clock readings
    k0 : int
        initial number of targets visited
    opt_policy : numpy array
        optimal policy

    Returns
    -------
    None.
    '''

    counter = 0
    while counter < 15:
        counter += 1
        print(f"Current location: {loc}, Current clock readings: {clk}, no. of targets visited: {k}")
        action = int(opt_policy[loc, np.where(np.all(clk == clock_space, axis = 1))[0][0], k])
        print("Optimal action: ", action)
        loc, clk, k = transition_fxn(loc, clk, k, action)
        print(f"New location: {loc}, Updated clock readings: {clk}, no. of targets visited: {k}")
        print('---------------------')
    return
    

if __name__ == "__main__":
    # loc_space contains the possible locations. 0 represents the depot and 1 to n represent the targets
    loc_space = np.arange(n + 1)

    # k_space contains the possible number of targets visited
    k_space = np.arange(K + 1)

    # clock_space contains the possible clock readings. At index 0 is the clock reading of the depot and at
    # index i is the clock reading of target i
    clock_space = combinations(n + 1, T)
    clock_space = np.array(clock_space)

    # action_space contains the possible actions, where action represents the location from loc_space to visit next.
    # 0 represents the depot and 1 to n represent the targets
    action_space = np.arange(n + 1)

    # initialize the q function to 0
    q = np.zeros((len(loc_space), len(clock_space), len(k_space), len(action_space)))
    q = q.reshape(-1, q.shape[-1])
    
    state_size = len(loc_space) * len(clock_space) * len(k_space)
    action_size = len(action_space)
    
    qtable_init = np.zeros((state_size, action_size))
    
    qtable, rewards = qlearning(qtable_init, 20000, 0.7, K, 0.95)
    print ("Score over time: " +  str(sum(rewards)/20000))
    print(qtable)

    # simulate the agent's walk for 20 steps
    loc0 = 0
    clk0 = np.zeros(n + 1)
    k0 = 0
    walk(loc0, clk0, k0, opt_policy)
    print("without GPU:", end-start)
