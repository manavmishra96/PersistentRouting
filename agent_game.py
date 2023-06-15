import gym
from gym import spaces
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

class SnakeGame(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

<<<<<<< HEAD:agent_game.py
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
=======
    def __init__(self, size=40):
        """
        Initialize the SnakeGame environment.

        Args:
            size (int): The size of the game grid.
        """
        super(SnakeGame, self).__init__()
        self.steps = 0
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py
        self.game_over = False
        self.size = size
        # Observation are dictionaries with the snake's and the apple's location.
        self.observation_space = spaces.Dict(
            {
                "snake": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "apple": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            dict: Dictionary containing the snake's and the apple's location.
        """
        return {"snake": self._snake_position, "apple": self._apple_position}

    def _get_info(self):
        """
        Get additional information about the environment.

        Returns:
            dict: Dictionary containing the distance between snake and apple.
        """
        return {"distance": np.linalg.norm(self._snake_position - self._apple_position, ord=1)}

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            dict: Initial observation of the environment.
        """
        self.game_over = False
        self._snake_position = np.array([self.size // 2, self.size // 2])
        self._apple_position = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])

        obs = self._get_obs()

        return obs

    def step(self, action):
<<<<<<< HEAD:agent_game.py
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
=======
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Tuple containing the next observation, reward, done flag, and additional information.
        """
        self.steps += 1
        if action == 0:
            self._snake_position[0] += 1  # right
        elif action == 1:
            self._snake_position[1] += 1  # up
        elif action == 2:
            self._snake_position[0] -= 1  # left
        elif action == 3:
            self._snake_position[1] -= 1  # down

        # Check if game is over and calculate the score
        if np.array_equal(self._snake_position, self._apple_position):
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py
            self.game_over = True
            score = 0.0
        elif (
            self._snake_position[0] >= self.size
            or self._snake_position[0] < 0
            or self._snake_position[1] >= self.size
            or self._snake_position[1] < 0
        ):
            self.game_over = True
            score = -50.0
        else:
            self.game_over = False
            score = self.manhattan_distance()
            score = -score

<<<<<<< HEAD:agent_game.py
        obs = self.render('rgb_array')
        reward = self.score
=======
        obs = self._get_obs()
        reward = float(score)
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py
        done = self.game_over
        info = self._get_info()

        return obs, reward, done, info

<<<<<<< HEAD:agent_game.py
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
=======
    def manhattan_distance(self):
        """
        Calculate the Manhattan distance between the snake and the apple.

        Returns:
            float: The Manhattan distance.
        """
        return float(np.sum(np.abs(self._snake_position - self._apple_position)))
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py

    def render(self, render_mode="human", folder="snake_simulation"):
        """
        Render the current state of the environment.

<<<<<<< HEAD:agent_game.py
if __name__ == '__main__':
    env = MonitorEnv()
=======
        Args:
            render_mode (str): The rendering mode ("human" or "rgb_array").
            folder (str): The folder to save the rendered images.
        """
        self.render_mode = render_mode
        if not self.game_over:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            img[self._snake_position[0], self._snake_position[1], :] = (0, 255, 0)
            img[self._apple_position[0], self._apple_position[1], :] = (255, 0, 0)
            if self.render_mode == "rgb_array":
                return img
            elif self.render_mode == "human":
                os.makedirs(folder, exist_ok=True)
                file_path = os.path.join(folder, f"snake_render_{self.steps}.png")
                plt.figure(figsize=(6, 6))
                img = np.rot90(img)
                plt.imshow(img, cmap="tab20", vmin=0, vmax=10)
                plt.axis("off")
                plt.savefig(file_path)
                plt.close()

if __name__ == "__main__":
    env = SnakeGame()
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py
    env.reset()
    total_reward = 0

    for i in range(150):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
<<<<<<< HEAD:agent_game.py
        print(obs)
        
=======
        env.render()
        print(f"Step: {i + 1}, Action: {action}, Reward: {reward}")
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")
>>>>>>> 7b3c0cd69c07b9fa1021c6acc06078056f7300cf:snake_game.py
    check_env(env)
