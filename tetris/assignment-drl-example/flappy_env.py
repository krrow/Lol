from __future__ import annotations
import math
import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    """
    Minimal Flappy Bird–like env for DRL testing demos.
    Uses simple rectangle collisions and scrolling pipes.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    WIDTH, HEIGHT = 400, 600
    PIPE_GAP = 140
    PIPE_SPEED = 3.2
    GRAVITY = 0.6
    FLAP_VEL = -8.5
    BIRD_X = 80
    BIRD_SIZE = 18
    PIPE_WIDTH = 55
    PIPE_SPAWN_EVERY = 90  # frames

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None,
                 reward_mode: str = "survival", max_steps: int = 5000):
        super().__init__()
        self.render_mode = render_mode
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.max_steps = max_steps

        # Observation: bird_y, bird_vel, next_pipe_x, gap_top, gap_bottom
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.reset(seed=seed)

        # Lazy pygame init
        self._pygame = None
        self._screen = None
        self._clock = None

    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self.bird_y = self.HEIGHT / 2
        self.bird_vel = 0.0
        self.frames = 0
        self.score = 0
        self._pipes = []  # list of dicts {x, gap_top}
        self._spawn_pipe(x=self.WIDTH + 40)
        self._spawn_pipe(x=self.WIDTH + 40 + 180)

        # Coverage: visited vertical bands (0..9)
        self._visited_bands = set()

        obs = self._get_obs()
        info = {"score": self.score, "bands_visited": 0}
        return obs, info

    def step(self, action: int):
        # Action
        if action == 1:
            self.bird_vel = self.FLAP_VEL
        else:
            self.bird_vel += self.GRAVITY

        # Clamp velocity
        self.bird_vel = float(np.clip(self.bird_vel, -12.0, 12.0))
        self.bird_y += self.bird_vel
        self.frames += 1

        # Move pipes left; spawn new ones periodically
        for p in self._pipes:
            p["x"] -= self.PIPE_SPEED
        if self.frames % self.PIPE_SPAWN_EVERY == 0:
            self._spawn_pipe(self.WIDTH + self.PIPE_WIDTH)

        # Remove offscreen pipes
        self._pipes = [p for p in self._pipes if p["x"] + self.PIPE_WIDTH > 0]

        # Scoring: passed pipe when its right edge crosses bird x
        for p in self._pipes:
            if not p.get("counted") and (p["x"] + self.PIPE_WIDTH) < self.BIRD_X:
                p["counted"] = True
                self.score += 1

        terminated = self._collided()
        truncated = self.frames >= self.max_steps

        reward = 0.0
        if self.reward_mode == "survival":
            reward += 0.1  # alive bonus
            reward += 1.0 * sum(1 for p in self._pipes if p.get("counted") and p.get("scored_frame") != self.frames)
        elif self.reward_mode == "coverage":
            reward += 0.05  # small alive bonus
            band = int(np.clip(math.floor(self.bird_y / (self.HEIGHT / 10)), 0, 9))
            if band not in self._visited_bands:
                self._visited_bands.add(band)
                reward += 0.05  # first time in a new vertical band
        else:
            reward += 0.1  # default alive bonus

        if terminated:
            reward -= 1.0

        obs = self._get_obs()
        info = {"score": self.score, "bands_visited": len(self._visited_bands)}
        
        if self.render_mode == "human":
            self._render_human()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            self._render_human()
        else:
            return None

    def close(self):
        if self._pygame:
            import pygame
            pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    # ---------------- Helpers ----------------
    def _spawn_pipe(self, x: float):
        margin = 60
        top = self._rnd.randint(margin, self.HEIGHT - margin - self.PIPE_GAP)
        self._pipes.append({"x": float(x), "gap_top": float(top), "counted": False})

    def _get_obs(self):
        next_pipe = min(self._pipes, key=lambda p: p["x"])  # leftmost
        gap_top = next_pipe["gap_top"]
        gap_bottom = gap_top + self.PIPE_GAP
        next_x = next_pipe["x"]

        obs = np.array([
            self.bird_y / self.HEIGHT,
            (self.bird_vel + 12.0) / 24.0,  # normalize to ~[0,1]
            next_x / self.WIDTH,
            gap_top / self.HEIGHT,
            gap_bottom / self.HEIGHT,
        ], dtype=np.float32)
        return obs

    def _collided(self) -> bool:
        # Ground / ceiling
        if self.bird_y < 0 or self.bird_y + self.BIRD_SIZE > self.HEIGHT:
            return True
        # Pipes
        bird_rect = (self.BIRD_X, self.bird_y, self.BIRD_SIZE, self.BIRD_SIZE)
        for p in self._pipes:
            # Upper pipe
            up_rect = (p["x"], 0, self.PIPE_WIDTH, p["gap_top"])
            # Lower pipe
            low_rect = (p["x"], p["gap_top"] + self.PIPE_GAP, self.PIPE_WIDTH,
                        self.HEIGHT - (p["gap_top"] + self.PIPE_GAP))
            if _rect_overlap(bird_rect, up_rect) or _rect_overlap(bird_rect, low_rect):
                return True
        return False

    # --------- Rendering helpers ---------
    def _lazy_pygame(self):
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            self._pygame.init()
            self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self._clock = pygame.time.Clock()

    def _render_human(self):
        self._lazy_pygame()
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self._draw_scene()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _render_rgb(self):
        self._lazy_pygame()
        self._draw_scene()
        import pygame
        arr = pygame.surfarray.array3d(self._screen)
        # transpose to HxWxC
        return np.transpose(arr, (1, 0, 2))

    def _draw_scene(self):
        pygame = self._pygame
        self._screen.fill((30, 30, 36))
        # Pipes
        for p in self._pipes:
            gap_top = int(p["gap_top"]) ; gap_bottom = gap_top + self.PIPE_GAP
            # upper
            pygame.draw.rect(self._screen, (80, 200, 120),
                             pygame.Rect(p["x"], 0, self.PIPE_WIDTH, gap_top))
            # lower
            pygame.draw.rect(self._screen, (80, 200, 120),
                             pygame.Rect(p["x"], gap_bottom, self.PIPE_WIDTH, self.HEIGHT-gap_bottom))
        # Bird
        pygame.draw.rect(self._screen, (230, 200, 40),
                         pygame.Rect(self.BIRD_X, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE))

def _rect_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by)