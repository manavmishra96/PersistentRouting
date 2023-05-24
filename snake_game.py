import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.env_checker import check_env

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'video']}

    def __init__(self, width=48, height=48, num_apples=5):
        super(SnakeEnv, self).__init__()
        self.steps = 0
        self.max_steps = 1000
        self.width = width
        self.height = height
        self.num_apples = num_apples

        self.snake_position = [self.width // 2, self.height // 2]
        self.snake_length = 1
        self.apple_positions = []

        for _ in range(self.num_apples):                                                                      # creates apple position randomly
            apple_pos = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            self.apple_positions.append(apple_pos)
        

        # self.initial_distance = -sum([abs(self.snake_position[0] - apple_pos[0]) +
        #                               abs(self.snake_position[1] - apple_pos[1]) 
        #                               for apple_pos in self.apple_positions])
        
        self.game_over = False

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        self.rendered_images = []
    


    def step(self, action):
        self.steps += 1
        if action == 0:
            self.snake_position[0] -= 1 #Snake moves left
        elif action == 1:
            self.snake_position[0] += 1 #Snake moves right
        elif action == 2:
            self.snake_position[1] -= 1 #Snake moves down
        elif action == 3:
            self.snake_position[1] += 1 #Snake moves up

        if len(self.apple_positions) != 0:
            if self.snake_position in self.apple_positions:
                self.apple_positions.remove(self.snake_position)
                # print(self.apple_positions)
                # print(len(self.apple_positions))


        # if self.snake_position in self.apple_positions:
        #     self.apple_positions.remove(self.snake_position)
        #     self.score += 1
        #     print(self.apple_positions)
        #     print(len(self.apple_positions))
        
        if self.snake_position[0] <= 0 or self.snake_position[0] >= self.width or \
                self.snake_position[1] <= 0 or self.snake_position[1] >= self.height:
            self.game_over = True
        
        if self.steps >= self.max_steps:
            self.game_over = True

        
        obs = self.render('rgb_array')
        reward = - sum([abs(self.snake_position[0] - apple_pos[0]) +
                                              abs(self.snake_position[1] - apple_pos[1]) for apple_pos in self.apple_positions])
        done = self.game_over
        info = {}

        return obs, reward, done, info

    def reset(self, num_apples=5):
        self.steps = 0
        self.num_apples = num_apples
        self.snake_position = [self.width // 2, self.height // 2]
        self.snake_length = 1
        
        self.apple_positions = []
        for _ in range(self.num_apples):
            apple_pos = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            self.apple_positions.append(apple_pos)
        self.initial_distance = -sum([abs(self.snake_position[0] - apple_pos[0]) +
                                      abs(self.snake_position[1] - apple_pos[1]) for apple_pos in self.apple_positions])
        self.game_over = False
        return self.render('rgb_array')

    def render(self, mode='human'):
        img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        img2d = np.zeros((self.width, self.height), dtype=np.uint8)
        if not self.game_over:
            img[self.snake_position[0], self.snake_position[1], :] = [0, 255, 0]    # Snake color: Green
            img2d[self.snake_position[0], self.snake_position[1]]  = 5  
            for apple_pos in self.apple_positions:
                img[apple_pos[0], apple_pos[1], :] = [255, 0, 0]  # Apple color: Red
                img2d[apple_pos[0], apple_pos[1]] = 10  

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.imshow('Snake Game', img)
            plt.show()
        elif mode == 'video':
            self.rendered_images.append(img2d)
            os.makedirs('images', exist_ok=True)
            for i, image in enumerate(self.rendered_images):
                plt.figure(figsize=(8, 8), dpi=80)
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(f'images/{i}.png')
                plt.close()




if __name__ == '__main__':
    env = SnakeEnv()
    env.reset()
    print(env.apple_positions)

    for step in range(1200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step: {step+1}, Action: {action}, Snake_position: {env.snake_position}, Reward: {reward}")
        if done:
            break
        # env.render('video')

    check_env(env)
