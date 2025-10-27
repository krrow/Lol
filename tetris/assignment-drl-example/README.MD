# FlappyBird DRL Mini‑Project (Gymnasium + Stable‑Baselines3)

A minimal, self‑contained example your students can run to learn DRL‑for‑testing with **Gymnasium**, **Stable‑Baselines3 (SB3)**, and **Python**. It includes:

* A tiny Flappy Bird–style environment (`FlappyBirdEnv`) with two reward modes — **survival** and **coverage** — to illustrate how reward design changes agent behavior/testing goals.
* Ready‑to‑run **training**, **evaluation**, and **(optional) visualization** scripts.
* Clear places to inject **faults/bugs** for testing objectives.

---

## 1) Quickstart

```bash
# 1) Create & activate a fresh venv (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install --upgrade pip
pip install gymnasium stable-baselines3[extra] pygame numpy tensorboard

# 3) Save the files below into the same folder, then train:
python train.py --timesteps 200000 --reward_mode survival --seed 7

# 4) Evaluate the trained agent (renders optional)
python eval.py --episodes 10 --render 0

# (Optional) Watch a live episode with rendering
python visualize.py --model_path models/ppo_flappy_survival --fps 60
```

**Folder layout (created at runtime):**

```
./
  flappy_env.py
  train.py
  eval.py
  visualize.py
  models/
  logs/
```

---

## 2) The Environment: `flappy_env.py`

Key ideas:

* Discrete(2) actions: **0=do nothing**, **1=flap**.
* Observation = `[bird_y, bird_vel, next_pipe_x, gap_top, gap_bottom]` (all normalized to \~\[0,1]).
* Termination on **collision**; truncation on **max\_steps** (episode cap).
* Two reward modes:

  * `survival`: +0.1 per step alive, +1.0 for each pipe passed, −1.0 on crash.
  * `coverage`: encourages exploring **new vertical bands** (simple testing proxy), +0.05 for first visit to a band in the episode, +0.05 per step alive, −1.0 on crash.


> **Where to inject faults (for testing):**
>
> * Make `PIPE_GAP` sometimes too small (e.g., 60) after N frames.
> * Randomly skip pipe collision for 1 frame.
> * Flip sign of `GRAVITY` for a short burst.
