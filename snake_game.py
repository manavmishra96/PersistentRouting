import gym
from gym import spaces
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

class SnakeGame(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size=40):
        """
        Initialize the SnakeGame environment.

        Args:
            size (int): The size of the game grid.
        """
        super(SnakeGame, self).__init__()
        self.steps = 0
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

        obs = self._get_obs()
        reward = float(score)
        done = self.game_over
        info = self._get_info()

        return obs, reward, done, info

    def manhattan_distance(self):
        """
        Calculate the Manhattan distance between the snake and the apple.

        Returns:
            float: The Manhattan distance.
        """
        return float(np.sum(np.abs(self._snake_position - self._apple_position)))

    def render(self, render_mode="human", folder="snake_simulation"):
        """
        Render the current state of the environment.

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
    env.reset()
    total_reward = 0

    for i in range(150):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Step: {i + 1}, Action: {action}, Reward: {reward}")
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")
    check_env(env)
