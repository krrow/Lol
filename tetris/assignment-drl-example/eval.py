# collects rich gameplay metrics
import argparse, os, csv
import numpy as np
from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv

def run_episode(model, reward_mode="survival", render=False):
    env = FlappyBirdEnv(render_mode="human" if render else None, reward_mode=reward_mode)
    obs, info = env.reset()
    done = trunc = False

    ep_reward = 0.0
    steps = 0
    clicks = 0  # number of flaps (action==1)

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        clicks += int(action == 1)
        obs, r, done, trunc, info = env.step(int(action))
        ep_reward += r
        steps += 1

    # Episode-level metrics from env info
    pipes_passed = int(info.get("score", 0))
    bands_visited = int(info.get("bands_visited", 0))  # for coverage mode

    env.close()
    return {
        "reward": float(ep_reward),
        "pipes_passed": pipes_passed,
        "steps": steps,
        "clicks": clicks,
        "crashed": int(done and not trunc),
        "truncated": int(trunc),
        "bands_visited": bands_visited,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/ppo_flappy_survival")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="survival", choices=["survival", "coverage"])
    p.add_argument("--csv_out", type=str, default="logs/eval_metrics.csv")
    args = p.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    model = PPO.load(args.model_path)

    rows = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, reward_mode=args.reward_mode, render=bool(args.render))
        metrics["episode"] = ep
        rows.append(metrics)

    # Summary
    mean_reward = float(np.mean([r["reward"] for r in rows]))
    std_reward  = float(np.std([r["reward"] for r in rows]))
    mean_pipes  = float(np.mean([r["pipes_passed"] for r in rows]))
    mean_clicks = float(np.mean([r["clicks"] for r in rows]))
    crash_rate  = float(np.mean([r["crashed"] for r in rows]))

    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean pipes passed: {mean_pipes:.2f}")
    print(f"Mean clicks (flaps): {mean_clicks:.2f}")
    print(f"Crash rate: {crash_rate*100:.1f}%")

    # Per-episode CSV
    fieldnames = ["episode","reward","pipes_passed","steps","clicks","crashed","truncated","bands_visited"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved metrics to {args.csv_out}")

if __name__ == "__main__":
    main()
