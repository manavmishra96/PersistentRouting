import numpy as np
from tqdm import tqdm

# fix the number of targets 'n' and the maximum number of visits 'K' in each walk
n = 2
K = 3

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
    

def value_iteration(v, delta, tol = 1e-6):
    '''
    performs value iteration to find the optimal value function and the optimal policy

    Parameters
    ----------
    v : numpy array
        value function
    delta : float
        discount factor
    tol : float, optional
        tolerance. The default is 1e-6.

    Returns
    -------
    v : numpy array
        optimal value function
    policy : numpy array
        optimal policy
    '''
    
    # initialize the policy matrix
    policy = np.zeros((len(loc_space), len(clock_space), len(k_space)))

    # set flag to 1 to signify that the value function has not converged
    flag = 1
    while flag == 1:
        flag = 0

        for loc in loc_space:
            for clk in range(len(clock_space)):
                for k in k_space:
                    best_value = np.inf
                    for a in action_space:
                        
                        # create a copy of current clock readings
                        times1, times2 = clock_space[clk].copy(), clock_space[clk].copy()

                        # get the new state
                        loc_new, clk_new, k_new = transition_fxn(loc, times1, k, a)

                        # get the reward
                        reward = reward_fxn(loc, times2, k, a)
                        
                        # calculate the present value
                        present_value = reward + delta*v[loc_new, np.where(np.all(clk_new == clock_space, axis = 1))[0][0], k_new]

                        # if the current action is giving lower value than the previous best action, update the best value and action
                        if present_value < best_value:
                            best_value = present_value
                            best_action = a
                    
                    # update the value function and policy
                    change = np.abs(best_value - v[loc, clk, k])
                    v[loc, clk, k] = best_value
                    policy[loc, clk, k] = best_action

                    error = np.min([tol, change])
                    # check convergence condition
                    if error >= tol:
                        flag = 1

    return v, policy

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
    while counter < 20:
        counter += 1
        print(f"Current location: {loc}, Current clock readings: {clk}, no. of targets visited: {k}")
        action = int(opt_policy[loc, np.where(np.all(clk == clock_space, axis = 1))[0][0], k])
        print("Optimal action: ", action)
        loc, clk, k = transition_fxn(loc, clk, k, action)
        print(f"New location: {loc}, Updated clock readings: {clk}, no. of targets visited: {k}")
        print('---------------------')
    return
    
# loc_space contains the possible locations. 0 represents the depot and 1 to n represent the targets
loc_space = np.arange(n + 1)

# k_space contains the possible number of targets visited
k_space = np.arange(K + 1)

# clock_space contains the possible clock readings. At index 0 is the clock reading of the depot and at
# index i is the clock reading of target i
clock_space = combinations(n + 1, K)
clock_space = np.array(clock_space)

# action_space contains the possible actions, where action represents the location from loc_space to visit next.
# 0 represents the depot and 1 to n represent the targets
action_space = np.arange(n + 1)

# initialize the value function to 0
v = np.zeros((len(loc_space), len(clock_space), len(k_space)))
opt_value, opt_policy = value_iteration(v, 0.9, tol = 1e-6)

# simulate the agent's walk for 20 steps
loc0 = 0
clk0 = np.zeros(n + 1)
k0 = 0
walk(loc0, clk0, k0, opt_policy)
