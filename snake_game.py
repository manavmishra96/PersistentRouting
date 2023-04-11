import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=40, height=40):
        super(SnakeEnv, self).__init__()
        self.width = width
        self.height = height

        self.snake_position = [self.width // 2, self.height // 2]
        self.snake_length = 1
        # self.snake_body = [self.snake_position]
        self.apple_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
        self.score = 0
        self.game_over = False

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        if action == 0:
            self.snake_position[0] -= 1
        elif action == 1:
            self.snake_position[0] += 1
        elif action == 2:
            self.snake_position[1] -= 1
        elif action == 3:
            self.snake_position[1] += 1

        if self.snake_position == self.apple_position:
            # self.snake_length += 1
            self.apple_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            self.score += 1

        # self.snake_body.append(self.snake_position)

        # if len(self.snake_body) > self.snake_length:
        #     del self.snake_body[0]

        if self.snake_position[0] < 0 or self.snake_position[0] >= self.width or \
                self.snake_position[1] < 0 or self.snake_position[1] >= self.height:
            self.game_over = True

        # if self.snake_position in self.snake_body[:-1]:
        #     self.game_over = True

        obs = self.render('rgb_array')
        reward = self.score
        done = self.game_over
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.snake_position = [self.width // 2, self.height // 2]
        self.snake_length = 1
        # self.snake_body = [self.snake_position]
        self.apple_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
        self.score = 0
        self.game_over = False
        return self.render('rgb_array')

    def render(self, mode='human'):
        img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        if self.game_over == False:
            img[self.snake_position[0], self.snake_position[1], :] = [0, 255, 0]
            img[self.apple_position[0], self.apple_position[1], :] = [255, 0, 0]
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from PIL import Image
            return Image.fromarray(img)


if __name__ == '__main__':
    env = SnakeEnv()
    env.reset()

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
    check_env(env)

