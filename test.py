import gym
import argparse
import os
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from snake_game import SnakeGame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ppo", help='ppo or a2c or dqn')
    return parser.parse_args()

# def save_video():


def evaluate_model(model, env):
    obs = env.reset()
    done = False
    total_reward = 0
    step_num = 0
    while not done:
        step_num += 1
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Step: {step_num}, Action: {action}, Reward: {reward}")
        env.render('human')
        # env.save_video()
        total_reward = reward
    return total_reward



if __name__ == "__main__":
    args = parse_args()

    env = SnakeGame(size=120)

    if args.model == "ppo":
        log_dir = "logs/ppo_logs/"
        env = Monitor(env, log_dir)
        ppo_model_path = os.path.join(log_dir, "best_model.zip")
        ppo_model = PPO.load(ppo_model_path)
        ppo_reward = evaluate_model(ppo_model, env)
        print(f"PPO Reward: {ppo_reward:.2f}")
    elif args.model == "a2c":
        log_dir = "logs/a2c_logs/"
        env = Monitor(env, log_dir)
        a2c_model_path = os.path.join(log_dir, "best_model.zip")
        a2c_model = A2C.load(a2c_model_path)
        a2c_reward = evaluate_model(a2c_model, env)
        print(f"A2C Reward: {a2c_reward:.2f}")
    elif args.model == "dqn":
        log_dir = "logs/dqn_logs/"
        env = Monitor(env, log_dir)
        dqn_model_path = os.path.join(log_dir, "best_model.zip")
        dqn_model = DQN.load(dqn_model_path)
        dqn_reward = evaluate_model(dqn_model, env)
        print(f"DQN Reward: {dqn_reward:.2f}")