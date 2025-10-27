import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from flappy_env import FlappyBirdEnv

def make_env(render_mode=None, reward_mode="survival", seed=7):
    env = FlappyBirdEnv(render_mode=render_mode, reward_mode=reward_mode, seed=seed)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--reward_mode", type=str, default="survival", choices=["survival", "coverage"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--modeldir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    env = make_env(reward_mode=args.reward_mode, seed=args.seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])  # nice to inspect in TB
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_name = f"ppo_flappy_{args.reward_mode}"
    path = os.path.join(args.modeldir, save_name)
    model.save(path)
    print(f"Saved model to {path}")

    env.close()


if __name__ == "__main__":
    main()