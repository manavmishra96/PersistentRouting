import gym
import os
import argparse
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from Snakega
import wandb
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


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--model', type=str, default="ppo", help='ppo or a2c or dqn')
   return parser.parse_args()

args = parse_args()

if args.model == "ppo":
  # PPO training
  wandb.init(project='Persistent Routing', name='PPO', sync_tensorboard=True)
  log_dir = "ppo_logs/"
  os.makedirs(log_dir, exist_ok=True)
  env = SnakeOneAppleEnv()
  env = Monitor(env, log_dir)
   
  print("PPO_run...")
  model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  model.learn(total_timesteps=5e6, tb_log_name="PPO_run", callback=callback, progress_bar=True)
  
  
elif args.model == "a2c":
  # A2C training
  wandb.init(project='Persistent Routing', name='A2C', sync_tensorboard=True)
  log_dir = "a2c_logs/"
  os.makedirs(log_dir, exist_ok=True)
  env = SnakeOneAppleEnv()
  env = Monitor(env, log_dir)

  print("A2C_run...")
  model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  model.learn(total_timesteps=5e6, tb_log_name="A2C_run", callback=callback, progress_bar=True)

# # model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)

elif args.model == "dqn":
  # DQN training
  wandb.init(project='Persistent Routing', name='DQN', sync_tensorboard=True)
  log_dir = "dqn_logs/"
  os.makedirs(log_dir, exist_ok=True)
  env = SnakeOneAppleEnv()
  env = Monitor(env, log_dir)

  print("DQN_run...")
  model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  model.learn(total_timesteps=5e6, tb_log_name="DQN_run", callback=callback, progress_bar=True)