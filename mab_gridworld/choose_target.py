import numpy as np
from mab_env import bandit_env
import random
import matplotlib.pyplot as plt

depot = (0, 0)
def generate_targets(n):
    targets = np.random.randint(10, size = (n, 2))
    present_ind = np.random.choice(range(len(targets)))
    present_loc = targets[present_ind, :]
    targets = np.delete(targets, present_ind, axis = 0)
    return present_loc, targets

present_loc, targets = generate_targets(6)

print("Present location: ", present_loc)
print("Targets: ", targets)

plt.plot(depot[0], depot[1], 'ro')
plt.plot(present_loc[0], present_loc[1], 'bo')
plt.plot(targets[:, 0], targets[:, 1], 'go')
plt.show(block = False)
plt.pause(3)
plt.close()

bandits = bandit_env(depot, present_loc, targets)

def random_pull():
    return np.random.choice(range(len(targets)))

true_means = abs(targets - present_loc).sum(axis = 1) + abs(targets - depot).sum(axis = 1)


def epsilon_greedy(epsilon, bandit):
    q = np.zeros(5)
    q_over_time = np.zeros((5, 1000))
    visited = np.zeros(len(targets))
    rewards = []
    for i in range(1000):
        if random.random() < epsilon:
            arm = random_pull()
        else:
            arm = np.argmin(q)
        
        visited[arm] += 1
        reward = bandit.pull(arm)
        rewards.append(reward)

        q[arm] += (reward - q[arm])/visited[arm]
        q_over_time[:, i] = q
    return q, rewards, q_over_time

q, reward, q_over_time = epsilon_greedy(0.1, bandit = bandits)

print("Q: ", q)
print("true means: ", true_means)
