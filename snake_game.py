import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env


class SnakeEnv(gym.Env):
    metadata = {'render modes': ['human']}

    def __init__(self, width=40, height=40, num_apples=1):
        super(SnakeEnv, self).__init__()
        self.height = height
        self.width = width
        self.num_apples = num_apples
        self.steps = 0
        self.game_over = False
        self.snake_position = [self.width // 2, self.height // 2]
        self.apple_position = self._generate_apple()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=40, shape=(3,), dtype=np.uint8)

    
    def reset(self):
        self.steps = 0
        self.game_over = False
        self.snake_position = [self.width // 2, self.height // 2]
        #code for generating apple
        self.apple_position = self._generate_apple()
        return self._get_states()
    
    def step(self, action):
        self.steps += 1
        if not self.game_over:
            if action == 0:
                self.snake_position[0] -= 1
            elif action == 1:
                self.snake_position[0] += 1
            elif action == 2:
                self.snake_position[1] -= 1
            elif action == 3:
                self.snake_position[1] += 1
        
        if self.snake_position[0] < 0 or self.snake_position[0] > self.width \
        or self.snake_position[1] < 0 or self.snake_position[1] > self.height\
            or len(self.apple_position) == 0:
            self.game_over = True
        
        obs = self._get_states()
        reward = -abs(self.snake_position[0] - self.apple_position[0][0]) - abs(self.snake_position[1] - self.apple_position[0][1])
        if self.snake_position == self.apple_position[0]:
            reward = 1
        done = self.game_over
        info = {}
        return obs, reward, done, info
    
    def _generate_apple(self):
        list_of_apples = []
        for _ in range(self.num_apples):
            apple_pos = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            list_of_apples.append(apple_pos)
        return list_of_apples
    
    # def _get_rewards(self):
    #     # if len(self.apple_position) == 0:
    #     #     # Reward for eating all apples (goal achieved)
    #     #     reward = 5
    #     if self.snake_position in self.apple_position:
    #         # Reward for eating an apple
    #         reward = 5
    #         self.apple_position.remove(self.snake_position)
    #     elif self.game_over:
    #         # Penalty for hitting the wall
    #         reward = -10
    #     else:
    #         # Small negative reward to encourage the agent to move and find apples
    #         reward = -0.01
    #     return reward

    
    def _get_states(self):
        distances = [abs(self.snake_position[0] - apple_pos[0]) + abs(self.snake_position[1] - apple_pos[1]) for apple_pos in self.apple_position]
        game_states = np.array([self.snake_position[0], self.snake_position[1], min(distances)], dtype=np.uint8)
        return game_states
        
    
    def render(self, mode='human', folder='snake_simulation'):
        img = np.zeros((self.width, self.height), dtype=np.uint8)
        if not self.game_over:
            img[self.snake_position[0]-1, self.snake_position[1]-1] = 5
            for apple_pos in self.apple_position:
                img[apple_pos[0]-1, apple_pos[1]-1] = 10
        if mode == 'human':
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, f"snake_render_{self.steps}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(file_path)
            plt.close()

if __name__ == '__main__':
    env = SnakeEnv()
    env.reset()
    total_reward = 0

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")
        total_reward += reward
        if done:
            break
        env.render('human')
    print(f"Total reward: {total_reward}")
    check_env(env)
