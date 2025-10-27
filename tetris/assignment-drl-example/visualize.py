import argparse
from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_flappy_survival")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    env = FlappyBirdEnv(render_mode="human")
    obs, info = env.reset()
    done, trunc = False, False

    # Run one episode with deterministic actions
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(action))
    env.close()


if __name__ == "__main__":
    main()