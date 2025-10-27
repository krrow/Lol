"""
Tetris Environment with Integrated Advanced Reward System

This is the complete TetrisEnv with the new reward system integrated.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Dict, Tuple, Optional, Any
from collections import deque, defaultdict


# Tetromino shapes and colors
SHAPES = [
    {'shape': [[1, 1, 1, 1]], 'name': 'I'},
    {'shape': [[1, 1], [1, 1]], 'name': 'O'},
    {'shape': [[1, 1, 1], [0, 1, 0]], 'name': 'T'},
    {'shape': [[1, 1, 1], [1, 0, 0]], 'name': 'L'},
    {'shape': [[1, 1, 1], [0, 0, 1]], 'name': 'J'},
    {'shape': [[1, 1, 0], [0, 1, 1]], 'name': 'S'},
    {'shape': [[0, 1, 1], [1, 1, 0]], 'name': 'Z'}
]

PIECE_COLORS = {
    'I': (0, 255, 255), 'O': (255, 255, 0), 'T': (128, 0, 128),
    'L': (255, 165, 0), 'J': (0, 0, 255), 'S': (0, 255, 0), 'Z': (255, 0, 0)
}


class TetrisRewardSystem:
    """Advanced reward system for Tetris DRL agents"""
    
    def __init__(self, play_style: str = 'balanced', grid_width: int = 10, grid_height: int = 20):
        self.play_style = play_style
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # State tracking
        self.prev_holes = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        self.prev_wells = 0
        self.prev_complete_lines = 0
        
        # Episode tracking
        self.pieces_since_clear = 0
        self.recent_line_clears = deque(maxlen=10)
        self.consecutive_clears = 0
        self.max_height_seen = 0
        self.dangerous_state_count = 0
        
        self.weights = self._get_style_weights(play_style)
    
    def _get_style_weights(self, style: str) -> Dict[str, float]:
        """Load reward weights from config file"""
        import yaml
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'persona_{style}.yaml')
        if not os.path.exists(config_path):
            style = 'balanced'
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'persona_{style}.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['persona_weights']
    
    def reset(self):
        """Reset for new episode"""
        self.prev_holes = 0
        self.prev_height = 0
        self.prev_bumpiness = 0
        self.prev_wells = 0
        self.prev_complete_lines = 0
        self.pieces_since_clear = 0
        self.recent_line_clears.clear()
        self.consecutive_clears = 0
        self.max_height_seen = 0
        self.dangerous_state_count = 0
    
    def calculate_reward(
        self,
        grid: np.ndarray,
        lines_cleared: int,
        piece_placed: bool,
        game_over: bool,
        action: int,
        valid_action: bool,
        piece_y: int
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward"""
        breakdown = {}
        
        # Calculate features
        features = self._calculate_features(grid)
        
        # Terminal rewards
        if game_over:
            breakdown['game_over'] = self.weights['game_over']
            return sum(breakdown.values()), breakdown
        
        # Line clear rewards
        if lines_cleared > 0:
            if lines_cleared == 1:
                breakdown['line_clear'] = self.weights['single_clear']
            elif lines_cleared == 2:
                breakdown['line_clear'] = self.weights['double_clear']
            elif lines_cleared == 3:
                breakdown['line_clear'] = self.weights['triple_clear']
            elif lines_cleared == 4:
                breakdown['line_clear'] = self.weights['tetris_clear']
            
            # Combo bonus
            self.consecutive_clears += 1
            if self.consecutive_clears > 1:
                breakdown['combo'] = (self.consecutive_clears - 1) * self.weights['combo_bonus']
            
            # Efficiency bonus
            if lines_cleared >= 2:
                breakdown['efficiency'] = (lines_cleared - 1) * self.weights['efficiency']
            
            self.pieces_since_clear = 0
            self.recent_line_clears.append(lines_cleared)
        else:
            self.consecutive_clears = 0
            self.pieces_since_clear += 1 if piece_placed else 0
        
        # Basic rewards
        breakdown['survival'] = self.weights['survival']
        if piece_placed:
            breakdown['piece_placed'] = self.weights['piece_placed']
        
        # Board state quality
        hole_diff = features['holes'] - self.prev_holes
        if hole_diff > 0:
            breakdown['holes'] = hole_diff * self.weights['hole_created']
        elif hole_diff < 0:
            breakdown['holes'] = abs(hole_diff) * self.weights['hole_filled']
        
        height_diff = features['max_height'] - self.prev_height
        if height_diff > 0:
            breakdown['height'] = height_diff * self.weights['height_increase']
        elif height_diff < 0:
            breakdown['height'] = abs(height_diff) * self.weights['height_decrease']
        
        bump_diff = features['bumpiness'] - self.prev_bumpiness
        if bump_diff > 0:
            breakdown['bumpiness'] = bump_diff * self.weights['bumpiness_increase']
        elif bump_diff < 0:
            breakdown['bumpiness'] = abs(bump_diff) * self.weights['bumpiness_decrease']
        
        well_diff = features['wells'] - self.prev_wells
        if well_diff > 0:
            breakdown['wells'] = well_diff * self.weights['well_created']
        elif well_diff < 0:
            breakdown['wells'] = abs(well_diff) * self.weights['well_filled']
        
        # Strategic rewards
        if features['almost_complete'] > self.prev_complete_lines and lines_cleared == 0:
            breakdown['line_setup'] = (features['almost_complete'] - self.prev_complete_lines) * self.weights['line_setup']
        
        if self._has_tetris_well(grid, features):
            breakdown['tetris_setup'] = self.weights['tetris_setup']
        
        # Danger penalties
        danger_threshold = self.weights['height_danger_threshold'] * self.grid_height
        if features['max_height'] > danger_threshold:
            danger_level = (features['max_height'] - danger_threshold) / self.grid_height
            breakdown['danger'] = danger_level * self.weights['danger_penalty']
            self.dangerous_state_count += 1
        
        # Action quality
        if not valid_action:
            breakdown['invalid'] = self.weights['invalid_action']
        
        if action == 5 and not piece_placed:  # No-op
            breakdown['waste'] = self.weights['waste_movement']
        
        # Position rewards
        if piece_placed and piece_y > self.grid_height / 2:
            breakdown['low_placement'] = self.weights['low_placement']
        
        if piece_placed and features['coverage_score'] > 0:
            breakdown['coverage'] = features['coverage_score'] * self.weights['coverage']
        
        # Update state
        self.prev_holes = features['holes']
        self.prev_height = features['max_height']
        self.prev_bumpiness = features['bumpiness']
        self.prev_wells = features['wells']
        self.prev_complete_lines = features['almost_complete']
        self.max_height_seen = max(self.max_height_seen, features['max_height'])
        
        total_reward = sum(breakdown.values())
        return total_reward, breakdown
    
    def _calculate_features(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate board features"""
        features = {}
        
        # Column heights
        heights = np.zeros(self.grid_width)
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                if grid[row, col]:
                    heights[col] = self.grid_height - row
                    break
        
        # Holes
        holes = 0
        for col in range(self.grid_width):
            block_found = False
            for row in range(self.grid_height):
                if grid[row, col]:
                    block_found = True
                elif block_found:
                    holes += 1
        features['holes'] = holes
        
        # Heights
        features['max_height'] = np.max(heights)
        features['avg_height'] = np.mean(heights)
        
        # Bumpiness
        bumpiness = 0
        for i in range(self.grid_width - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        features['bumpiness'] = bumpiness
        
        # Wells
        wells = 0
        well_depths = []
        for i in range(self.grid_width):
            left_higher = (i == 0) or (heights[i - 1] > heights[i])
            right_higher = (i == self.grid_width - 1) or (heights[i + 1] > heights[i])
            if left_higher and right_higher:
                wells += 1
                left_h = heights[i - 1] if i > 0 else float('inf')
                right_h = heights[i + 1] if i < self.grid_width - 1 else float('inf')
                depth = min(left_h, right_h) - heights[i]
                if depth > 0:
                    well_depths.append(depth)
        features['wells'] = wells
        features['max_well_depth'] = max(well_depths) if well_depths else 0
        
        # Almost complete lines
        almost_complete = 0
        for row in range(self.grid_height):
            filled = np.sum(grid[row])
            if self.grid_width - 2 <= filled < self.grid_width:
                almost_complete += 1
        features['almost_complete'] = almost_complete
        
        # Coverage score
        filled_cells = np.sum(grid)
        if filled_cells > 0:
            height_std = np.std(heights)
            coverage_score = 1.0 / (1.0 + height_std)
        else:
            coverage_score = 0.0
        features['coverage_score'] = coverage_score
        
        return features
    
    def _has_tetris_well(self, grid: np.ndarray, features: Dict) -> bool:
        """Check if board has a good Tetris well setup"""
        if features['max_well_depth'] < 4:
            return False
        
        heights = np.zeros(self.grid_width)
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                if grid[row, col]:
                    heights[col] = self.grid_height - row
                    break
        
        # Check edges for deep wells
        left_well = heights[1] - heights[0] >= 4 if self.grid_width > 1 else False
        right_well = heights[-2] - heights[-1] >= 4 if self.grid_width > 1 else False
        
        return left_well or right_well
    
    def get_statistics(self) -> Dict:
        """Get episode statistics"""
        return {
            'pieces_since_clear': self.pieces_since_clear,
            'consecutive_clears': self.consecutive_clears,
            'recent_clears': list(self.recent_line_clears),
            'avg_recent_clear': np.mean(self.recent_line_clears) if self.recent_line_clears else 0,
            'max_height_seen': self.max_height_seen,
            'dangerous_state_count': self.dangerous_state_count,
        }


class TetrisEnv(gym.Env):
    """
    Tetris Environment with Integrated Advanced Reward System
    
    Actions:
        0: Move left
        1: Move right
        2: Rotate clockwise
        3: Soft drop (move down)
        4: Hard drop
        5: No-op (do nothing)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid_width: int = 10,
        grid_height: int = 20,
        max_steps: int = 10000,
        gravity_interval: int = 10,
        play_style: str = 'balanced',
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gravity_interval = gravity_interval
        self.play_style = play_style
        
        # Action space
        self.action_space = spaces.Discrete(6)
        
        # Observation space
        obs_size = (grid_height * grid_width) + 7 + 7 + 2 + 10
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize reward system
        self.reward_system = TetrisRewardSystem(
            play_style=play_style,
            grid_width=grid_width,
            grid_height=grid_height
        )
        
        # Initialize pygame for rendering
        if self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            self.block_size = 30
            self.window_width = self.block_size * (self.grid_width + 8)
            self.window_height = self.block_size * self.grid_height
            self.screen = pygame.Surface((self.window_width, self.window_height))
            if self.render_mode == "human":
                self.display = pygame.display.set_mode((self.window_width, self.window_height))
                pygame.display.set_caption("Tetris DRL Environment")
            self.clock = pygame.time.Clock()
        
        # Metrics tracking
        self.reset_metrics()
        
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: int):
        """Set random seed"""
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
    
    def reset_metrics(self):
        """Reset all episode metrics"""
        self.metrics = {
            'lines_cleared': 0,
            'single_clears': 0,
            'double_clears': 0,
            'triple_clears': 0,
            'tetris_clears': 0,
            'total_pieces': 0,
            'actions_taken': 0,
            'holes_created': 0,
            'max_height': 0,
            'avg_height': 0,
            'total_bumpiness': 0,
            'wells_created': 0,
            'rotations': 0,
            'left_moves': 0,
            'right_moves': 0,
            'soft_drops': 0,
            'hard_drops': 0,
            'no_ops': 0,
            'invalid_actions': 0,
            'survival_time': 0,
            'coverage': 0,
            'piece_distribution': {name: 0 for name in ['I', 'O', 'T', 'L', 'J', 'S', 'Z']},
            'max_combo': 0,
            'current_combo': 0,
            'state_revisits': 0,
            'unique_states': 0,
        }
        self.visited_cells = set()
        self.state_history_set = set()
        self.state_history_deque = deque(maxlen=1000)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed is not None:
            self.seed(seed)
        
        super().reset(seed=seed)
        
        # Initialize game state
        self.bag = []
        self._refill_bag()
        
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self._final_frame_surface = None
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        self.game_over = False
        self.score = 0
        self.steps = 0
        self.piece_steps = 0
        self.piece_placed = False
        
        # Reset metrics and reward system
        self.reset_metrics()
        self.reward_system.reset()
        
        return self._get_observation(), {}
    
    def _new_piece(self) -> Dict:
        """Generate new piece using 7-bag"""
        if not hasattr(self, 'bag') or not self.bag:
            self._refill_bag()
        piece = self.bag.pop()
        return {
            'shape': np.array(piece['shape'], dtype=np.int8),
            'name': piece['name'],
            'x': self.grid_width // 2 - len(piece['shape'][0]) // 2,
            'y': 0,
            'color': PIECE_COLORS[piece['name']]
        }
    
    def _refill_bag(self):
        """Fill and shuffle 7-bag"""
        self.bag = SHAPES.copy()
        random.shuffle(self.bag)
    
    def _valid_move(self, piece: Dict, x: int, y: int, shape: Optional[np.ndarray] = None) -> bool:
        """Check if move is valid"""
        if shape is None:
            shape = piece['shape']
        
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    new_x = x + j
                    new_y = y + i
                    if (new_x < 0 or new_x >= self.grid_width or 
                        new_y >= self.grid_height or 
                        (new_y >= 0 and self.grid[new_y, new_x])):
                        return False
        return True
    
    def _place_piece(self) -> int:
        """Place piece on grid and return lines cleared"""
        # Place piece on grid
        for i in range(self.current_piece['shape'].shape[0]):
            for j in range(self.current_piece['shape'].shape[1]):
                if self.current_piece['shape'][i, j]:
                    y = self.current_piece['y'] + i
                    x = self.current_piece['x'] + j
                    if 0 <= y < self.grid_height and 0 <= x < self.grid_width:
                        self.grid[y, x] = 1
                        self.visited_cells.add((y, x))
        
        # Clear complete lines
        lines_cleared = 0
        for i in range(self.grid_height):
            if np.all(self.grid[i]):
                self.grid = np.vstack([np.zeros((1, self.grid_width), dtype=np.int8), 
                                       np.delete(self.grid, i, axis=0)])
                lines_cleared += 1
        
        # Update score
        if lines_cleared == 1:
            self.score += 40
            self.metrics['single_clears'] += 1
        elif lines_cleared == 2:
            self.score += 100
            self.metrics['double_clears'] += 1
        elif lines_cleared == 3:
            self.score += 300
            self.metrics['triple_clears'] += 1
        elif lines_cleared == 4:
            self.score += 1200
            self.metrics['tetris_clears'] += 1
        
        self.metrics['lines_cleared'] += lines_cleared
        
        if lines_cleared > 0:
            self.metrics['current_combo'] += 1
            self.metrics['max_combo'] = max(self.metrics['max_combo'], self.metrics['current_combo'])
        else:
            self.metrics['current_combo'] = 0
        
        # Get new piece
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()
        self.metrics['total_pieces'] += 1
        self.metrics['piece_distribution'][self.current_piece['name']] += 1
        self.piece_steps = 0
        
        # Check game over
        if not self._valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
            self.game_over = True
            if self.render_mode in ["human", "rgb_array"]:
                try:
                    self._final_frame_surface = self._render_static_frame()
                except Exception:
                    self._final_frame_surface = None
        
        return lines_cleared
    
    def _get_aggregate_features(self) -> np.ndarray:
        """Calculate aggregate features (normalized)"""
        features = np.zeros(10, dtype=np.float32)
        
        heights = np.zeros(self.grid_width)
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                if self.grid[row, col]:
                    heights[col] = self.grid_height - row
                    break
        
        # Holes
        holes = 0
        for col in range(self.grid_width):
            block_found = False
            for row in range(self.grid_height):
                if self.grid[row, col]:
                    block_found = True
                elif block_found:
                    holes += 1
        features[0] = holes / (self.grid_width * self.grid_height)
        
        # Heights
        features[1] = np.max(heights) / self.grid_height
        features[2] = np.mean(heights) / self.grid_height
        
        # Bumpiness
        bumpiness = 0
        for i in range(self.grid_width - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        features[3] = bumpiness / (self.grid_height * (self.grid_width - 1))
        
        # Wells
        wells = 0
        for i in range(self.grid_width):
            left_higher = i == 0 or heights[i - 1] > heights[i]
            right_higher = i == self.grid_width - 1 or heights[i + 1] > heights[i]
            if left_higher and right_higher:
                wells += 1
        features[4] = wells / self.grid_width
        
        # Complete lines
        complete_lines = 0
        for row in range(self.grid_height):
            if np.sum(self.grid[row]) >= self.grid_width - 1:
                complete_lines += 1
        features[5] = complete_lines / self.grid_height
        
        # Height quartiles
        sorted_heights = np.sort(heights)
        features[6] = sorted_heights[self.grid_width // 4] / self.grid_height
        features[7] = sorted_heights[self.grid_width // 2] / self.grid_height
        features[8] = sorted_heights[3 * self.grid_width // 4] / self.grid_height
        features[9] = sorted_heights[-1] / self.grid_height
        
        return features
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        grid_flat = self.grid.flatten().astype(np.float32)
        
        current_piece_encoding = np.zeros(7, dtype=np.float32)
        next_piece_encoding = np.zeros(7, dtype=np.float32)
        piece_names = ['I', 'O', 'T', 'L', 'J', 'S', 'Z']
        current_piece_encoding[piece_names.index(self.current_piece['name'])] = 1
        next_piece_encoding[piece_names.index(self.next_piece['name'])] = 1
        
        position = np.array([
            self.current_piece['x'] / self.grid_width,
            self.current_piece['y'] / self.grid_height
        ], dtype=np.float32)
        
        aggregate = self._get_aggregate_features()
        
        obs = np.concatenate([
            grid_flat,
            current_piece_encoding,
            next_piece_encoding,
            position,
            aggregate
        ])
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        self.piece_steps += 1
        self.metrics['actions_taken'] += 1
        self.metrics['survival_time'] = self.steps
        self.piece_placed = False
        
        valid_action = True
        lines_cleared = 0
        
        # Execute action
        if action == 0:  # Move left
            self.metrics['left_moves'] += 1
            if self._valid_move(self.current_piece, self.current_piece['x'] - 1, self.current_piece['y']):
                self.current_piece['x'] -= 1
            else:
                valid_action = False
                self.metrics['invalid_actions'] += 1
                
        elif action == 1:  # Move right
            self.metrics['right_moves'] += 1
            if self._valid_move(self.current_piece, self.current_piece['x'] + 1, self.current_piece['y']):
                self.current_piece['x'] += 1
            else:
                valid_action = False
                self.metrics['invalid_actions'] += 1
                
        elif action == 2:  # Rotate
            self.metrics['rotations'] += 1
            rotated = np.rot90(self.current_piece['shape'], k=-1)
            if self._valid_move(self.current_piece, self.current_piece['x'], 
                              self.current_piece['y'], rotated):
                self.current_piece['shape'] = rotated
            else:
                valid_action = False
                self.metrics['invalid_actions'] += 1
                
        elif action == 3:  # Soft drop
            self.metrics['soft_drops'] += 1
            if self._valid_move(self.current_piece, self.current_piece['x'], 
                              self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
            else:
                lines_cleared = self._place_piece()
                self.piece_placed = True
                
        elif action == 4:  # Hard drop
            self.metrics['hard_drops'] += 1
            while self._valid_move(self.current_piece, self.current_piece['x'], 
                                 self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
            lines_cleared = self._place_piece()
            self.piece_placed = True
                
        elif action == 5:  # No-op
            self.metrics['no_ops'] += 1
        
        # Natural gravity
        if not self.piece_placed and self.piece_steps % self.gravity_interval == 0:
            if self._valid_move(self.current_piece, self.current_piece['x'], 
                              self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
            else:
                lines_cleared = self._place_piece()
                self.piece_placed = True
        
        # Update metrics
        current_features = self._get_aggregate_features()
        self.metrics['holes_created'] = int(current_features[0] * (self.grid_width * self.grid_height))
        self.metrics['max_height'] = int(current_features[1] * self.grid_height)
        self.metrics['avg_height'] = float(current_features[2] * self.grid_height)
        self.metrics['total_bumpiness'] = int(current_features[3] * self.grid_height * (self.grid_width - 1))
        self.metrics['wells_created'] = int(current_features[4] * self.grid_width)
        self.metrics['coverage'] = len(self.visited_cells) / (self.grid_width * self.grid_height)
        
        # Track state revisits
        state_hash = hash(self.grid.tobytes())
        if state_hash in self.state_history_set:
            self.metrics['state_revisits'] += 1
        else:
            self.metrics['unique_states'] += 1
            self.state_history_set.add(state_hash)
        self.state_history_deque.append(state_hash)
        
        # Calculate reward using reward system
        reward, breakdown = self.reward_system.calculate_reward(
            grid=self.grid,
            lines_cleared=lines_cleared,
            piece_placed=self.piece_placed,
            game_over=self.game_over,
            action=action,
            valid_action=valid_action,
            piece_y=self.current_piece['y']
        )
        
        # Check termination
        terminated = self.game_over
        truncated = self.steps >= self.max_steps
        
        # Get info
        info = self._get_info()
        info['reward_breakdown'] = breakdown
        info['reward_stats'] = self.reward_system.get_statistics()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Return episode metrics"""
        return {
            'score': self.score,
            'steps': self.steps,
            **self.metrics
        }
    
    def _render_static_frame(self) -> pygame.Surface:
        """Render current board to surface"""
        surface = pygame.Surface((self.window_width, self.window_height))
        surface.fill((0, 0, 0))
        
        border_width = 4
        pygame.draw.rect(surface, (255, 255, 255),
                         (0, 0, self.grid_width * self.block_size + 2 * border_width,
                          self.window_height), border_width)
        
        # Draw grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j]:
                    pygame.draw.rect(surface, (100, 100, 255),
                                     (border_width + j * self.block_size,
                                      i * self.block_size,
                                      self.block_size - 1, self.block_size - 1))
        
        # Draw current piece
        for i in range(self.current_piece['shape'].shape[0]):
            for j in range(self.current_piece['shape'].shape[1]):
                if self.current_piece['shape'][i, j]:
                    pygame.draw.rect(surface, self.current_piece['color'],
                                     (border_width + (self.current_piece['x'] + j) * self.block_size,
                                      (self.current_piece['y'] + i) * self.block_size,
                                      self.block_size - 1, self.block_size - 1))
        
        # Draw info
        font = pygame.font.Font(None, 24)
        info_x = self.grid_width * self.block_size + border_width * 3
        texts = [
            f"Score: {self.score}",
            f"Lines: {self.metrics['lines_cleared']}",
            f"Pieces: {self.metrics['total_pieces']}",
            f"Steps: {self.steps}"
        ]
        for idx, text in enumerate(texts):
            surface.blit(font.render(text, True, (255, 255, 255)), (info_x, self.block_size * (idx + 1)))
        
        return surface
    
    def render(self):
        """Render the environment"""
        if self.render_mode not in ["human", "rgb_array"]:
            return
        
        border_width = 4
        
        if self.game_over:
            if getattr(self, '_final_frame_surface', None) is not None:
                self.screen.blit(self._final_frame_surface, (0, 0))
            else:
                self.screen.fill((0, 0, 0))
                pygame.draw.rect(self.screen, (255, 255, 255),
                                 (0, 0, self.grid_width * self.block_size + 2 * border_width,
                                  self.window_height), border_width)
                for i in range(self.grid_height):
                    for j in range(self.grid_width):
                        if self.grid[i, j]:
                            pygame.draw.rect(self.screen, (100, 100, 255),
                                             (border_width + j * self.block_size,
                                              i * self.block_size,
                                              self.block_size - 1, self.block_size - 1))
            
            # Game over overlay
            overlay = pygame.Surface((self.window_width, self.window_height))
            overlay.set_alpha(160)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            font = pygame.font.Font(None, 48)
            go_text = font.render('Game Over!', True, (255, 255, 255))
            score_text = font.render(f'Final Score: {self.score}', True, (255, 255, 255))
            hint_text = pygame.font.Font(None, 24).render('Call env.reset() to restart', True, (255, 255, 255))
            
            go_rect = go_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 40))
            score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            hint_rect = hint_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 40))
            
            self.screen.blit(go_text, go_rect)
            self.screen.blit(score_text, score_rect)
            self.screen.blit(hint_text, hint_rect)
            
            if self.render_mode == "human":
                self.display.blit(self.screen, (0, 0))
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            elif self.render_mode == "rgb_array":
                return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
            return
        
        # Normal rendering
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (0, 0, self.grid_width * self.block_size + 2 * border_width,
                          self.window_height), border_width)
        
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j]:
                    pygame.draw.rect(self.screen, (100, 100, 255),
                                     (border_width + j * self.block_size,
                                      i * self.block_size,
                                      self.block_size - 1, self.block_size - 1))
        
        for i in range(self.current_piece['shape'].shape[0]):
            for j in range(self.current_piece['shape'].shape[1]):
                if self.current_piece['shape'][i, j]:
                    pygame.draw.rect(self.screen, self.current_piece['color'],
                                     (border_width + (self.current_piece['x'] + j) * self.block_size,
                                      (self.current_piece['y'] + i) * self.block_size,
                                      self.block_size - 1, self.block_size - 1))
        
        font = pygame.font.Font(None, 24)
        info_x = self.grid_width * self.block_size + border_width * 3
        texts = [
            f"Score: {self.score}",
            f"Lines: {self.metrics['lines_cleared']}",
            f"Pieces: {self.metrics['total_pieces']}",
            f"Steps: {self.steps}"
        ]
        for idx, text in enumerate(texts):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (info_x, self.block_size * (idx + 1)))
        
        if self.render_mode == "human":
            self.display.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    
    def close(self):
        """Clean up resources"""
        if self.render_mode == "human":
            pygame.quit()


# Example usage
if __name__ == "__main__":
    # Test the environment
    env = TetrisEnv(render_mode="human", play_style='balanced')
    
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Play style: {env.play_style}")
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 1000:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: Reward={reward:.2f}, Total={total_reward:.2f}, Lines={info['lines_cleared']}")
            if 'reward_breakdown' in info:
                print(f"  Breakdown: {info['reward_breakdown']}")
        
        env.render()
    
    print(f"\nEpisode finished!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total steps: {step_count}")
    print(f"Lines cleared: {info['lines_cleared']}")
    print(f"Score: {info['score']}")
    
    env.close()