import gym
import os
from stable_baselines3 import A2C, DDPG, TD3, SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from snake_game import SnakeEnv
import numpy as np
import tensorboard

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


# PPO training
log_dir = "ppo_logs/"
os.makedirs(log_dir, exist_ok=True)
env = SnakeEnv()
env = Monitor(env, log_dir)

print("PPO_run...")
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=5e6, tb_log_name="PPO_run", callback=callback, progress_bar=True)


# A2C training
log_dir = "a2c_logs/"
os.makedirs(log_dir, exist_ok=True)
env = SnakeEnv()
env = Monitor(env, log_dir)

print("A2C_run...")
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=5e6, tb_log_name="A2C_run", callback=callback, progress_bar=True)

# # model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)

# DDPG training
log_dir = "ddpg_logs/"
os.makedirs(log_dir, exist_ok=True)
env = SnakeEnv()
env = Monitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("DDPG_run...")
model = DDPG("CnnPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="./tb_logs/")
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=5e6, tb_log_name="DDPG_run", callback=callback, progress_bar=True)


# TD3 training
log_dir = "td3_logs/"
os.makedirs(log_dir, exist_ok=True)
env = SnakeEnv()
env = Monitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("TD3_run...")
model = TD3("CnnPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="./tb_logs/")
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=5e6, tb_log_name="TD3_run", callback=callback, progress_bar=True)


# SAC training
log_dir = "sac_logs/"
os.makedirs(log_dir, exist_ok=True)
env = SnakeEnv()
env = Monitor(env, log_dir)

print("SAC_run...")
model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=5e6, tb_log_name="SAC_run", callback=callback, progress_bar=True)


