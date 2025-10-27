
"""Visualize a single deterministic episode of a trained agent playing
Tetris.

This script mirrors the simple `visualize.py` used for the Flappy demo and
loads a Stable-Baselines3 model (PPO by default). It opens the environment
with `render_mode='human'` and runs one episode with deterministic actions,
so the agent appears as a bot playing the game.

Usage example:
	python src/visualize.py --model_path models/ppo_survivor_seed7.zip \
		--algo ppo --persona survivor --seed 7 --fps 60
"""

import argparse
import time
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
	from stable_baselines3 import PPO, A2C
except Exception:  # pragma: no cover - best-effort import
	PPO = None
	A2C = None

# Add env directory to Python path
current_dir = Path(__file__).parent.absolute()
env_dir = current_dir.parent / 'env'
sys.path.insert(0, str(env_dir))

from tetris_env import TetrisEnv


def load_model(path: str, algo: str = "ppo"):
	"""Load a Stable-Baselines3 model. Default algo is PPO.

	Keeps things simple: caller should provide the right algo name that matches
	the saved model format ("ppo" or "a2c").
	"""
	if algo.lower() == "ppo":
		if PPO is None:
			raise RuntimeError("stable_baselines3.PPO is not available")
		return PPO.load(path)
	elif algo.lower() == "a2c":
		if A2C is None:
			raise RuntimeError("stable_baselines3.A2C is not available")
		return A2C.load(path)
	else:
		raise ValueError(f"Unsupported algo: {algo}")


def run_one_episode(model, persona: str = "survivor", seed: Optional[int] = None, fps: int = 60):
	"""Run a single episode with deterministic actions and render it.

	Returns the final info dict from the environment.
	"""
	env = TetrisEnv(render_mode="human", seed=seed)

	# Gym/Gymnasium reset differences: reset may return (obs, info)
	reset_res = env.reset()
	if isinstance(reset_res, tuple) and len(reset_res) == 2:
		obs = reset_res[0]
	else:
		obs = reset_res

	done = False
	truncated = False
	last_info = {}

	# If env exposes metadata.render_fps, prefer that; otherwise use provided fps
	try:
		fps = int(getattr(env, "metadata", {}).get("render_fps", fps))
	except Exception:
		fps = fps

	while True:
		action, _ = model.predict(obs, deterministic=True)
		print(f"Model action: {action} ({['LEFT', 'RIGHT', 'ROTATE', 'SOFT_DROP', 'HARD_DROP', 'NO_OP'][int(action)]})")
		step_res = env.step(int(action))

		# Support both gym (obs, reward, done, info) and gymnasium
		if len(step_res) == 5:
			obs, reward, terminated, truncated, info = step_res
			done = bool(terminated or truncated)
		else:
			obs, reward, done, info = step_res

		# Render frame (TetrisEnv uses pygame; render() is a no-op for non-human)
		try:
			env.render()
		except Exception:
			# Rendering should not crash the rollout; ignore render errors.
			pass

		last_info = info

		if done:
			break

		# Pace the loop. TetrisEnv.render() uses pygame.Clock.tick, but adding
		# a small sleep ensures consistent timing when render doesn't
		# throttle.
		time.sleep(1.0 / max(1, fps))

	# Save final info and close the env
	env.close()
	return last_info


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str, required=True,
						help="Path to the saved Stable-Baselines3 model (.zip)")
	parser.add_argument("--algo", type=str, choices=["ppo", "a2c"], default="ppo",
						help="Algorithm used to train the model (default: ppo)")
	parser.add_argument("--persona", type=str, choices=["survivor", "speedrunner"], default="survivor",
						help="Reward/persona used by the environment (default: survivor)")
	parser.add_argument("--seed", type=int, default=None, help="Optional seed for the env")
	parser.add_argument("--fps", type=int, default=60, help="Render FPS (default: 60)")
	args = parser.parse_args()

	if not os.path.exists(args.model_path):
		raise FileNotFoundError(f"Model not found: {args.model_path}")

	model = load_model(args.model_path, algo=args.algo)

	print(f"Running one deterministic episode using model: {args.model_path}")
	info = run_one_episode(model, persona=args.persona, seed=args.seed, fps=args.fps)

	print("Episode finished. Info:")
	print(info)


if __name__ == "__main__":
	main()
