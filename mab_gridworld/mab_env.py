import numpy as np
class bandit_env():

    def __init__(self, depot, present_loc, targets):
        self.depot = depot
        self.present_loc = present_loc
        self.targets = targets

        self.dist_to_targets = abs(self.targets - self.present_loc).sum(axis = 1)
        self.dist_to_depot = abs(self.targets - self.depot).sum(axis = 1)

    def pull(self, arm):
        return np.random.normal(self.dist_to_targets[arm] + self.dist_to_depot[arm], 1)