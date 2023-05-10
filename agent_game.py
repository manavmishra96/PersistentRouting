import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env


class MonitorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=40, height=40):
        super(MonitorEnv, self).__init__()
        
        # Define game height and width
        self.width = width
        self.height = height

        # Agent position and target position
        self.agent_position = [self.width // 2, self.height // 2]
        self.target_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
        
        self.score = 0
        self.game_over = False

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        # LEFT = 0, RIGHT = 1, DOWN = 2, UP = 3
        if action == 0:
            self.agent_position[0] -= 1
        elif action == 1:
            self.agent_position[0] += 1
        elif action == 2:
            self.agent_position[1] -= 1
        elif action == 3:
            self.agent_position[1] += 1

        if self.agent_position == self.target_position:
            self.target_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            self.score += 1

        if self.agent_position[0] < 0 or self.agent_position[0] >= self.width or \
                self.agent_position[1] < 0 or self.agent_position[1] >= self.height:
            self.game_over = True

        obs = self.render('rgb_array')
        reward = self.score
        done = self.game_over
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.agent_position = [self.width // 2, self.height // 2]
        self.target_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
        self.score = 0
        self.game_over = False
        return self.render('rgb_array')

    def render(self, mode='human'):
        img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        if self.game_over == False:
            img[self.agent_position[0], self.agent_position[1], :] = [0, 255, 0]
            img[self.target_position[0], self.target_position[1], :] = [255, 0, 0]
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from PIL import Image
            return Image.fromarray(img)


if __name__ == '__main__':
    env = MonitorEnv()
    env.reset()

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        
    check_env(env)

