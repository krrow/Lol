import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Gymnasium and Stable-Baselines3
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


class TetrisMetricsCallback(BaseCallback):
    """
    Custom callback for collecting detailed Tetris metrics during training.
    Tracks game-specific metrics like lines cleared, piece usage, and survival time.
    """
    
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Episode metrics
        self.episode_data = []
        
        # Aggregated metrics
        self.total_episodes = 0
        self.running_scores = []
        self.running_lines = []
        
    def _on_step(self) -> bool:
        """Called at each environment step"""
        # Check if episode ended
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            infos = self.locals['infos']
            
            for i, done in enumerate(dones):
                if done:
                    self._log_episode(infos[i])
        
        return True
    
    def _log_episode(self, info: Dict):
        """Log metrics for a completed episode"""
        episode_data = {
            'timestep': self.num_timesteps,
            'episode_num': self.total_episodes,
            'score': info.get('score', 0),
            'steps': info.get('steps', 0),
            'survival_time': info.get('survival_time', 0),
            
            # Line clearing metrics
            'lines_cleared': info.get('lines_cleared', 0),
            'single_clears': info.get('single_clears', 0),
            'double_clears': info.get('double_clears', 0),
            'triple_clears': info.get('triple_clears', 0),
            'tetris_clears': info.get('tetris_clears', 0),
            
            # Piece metrics
            'total_pieces': info.get('total_pieces', 0),
            'piece_I': info.get('piece_distribution', {}).get('I', 0),
            'piece_O': info.get('piece_distribution', {}).get('O', 0),
            'piece_T': info.get('piece_distribution', {}).get('T', 0),
            'piece_L': info.get('piece_distribution', {}).get('L', 0),
            'piece_J': info.get('piece_distribution', {}).get('J', 0),
            'piece_S': info.get('piece_distribution', {}).get('S', 0),
            'piece_Z': info.get('piece_distribution', {}).get('Z', 0),
            
            # Action metrics
            'actions_taken': info.get('actions_taken', 0),
            'rotations': info.get('rotations', 0),
            'left_moves': info.get('left_moves', 0),
            'right_moves': info.get('right_moves', 0),
            'soft_drops': info.get('soft_drops', 0),
            'hard_drops': info.get('hard_drops', 0),
            'no_ops': info.get('no_ops', 0),
            'invalid_actions': info.get('invalid_actions', 0),
            
            # State metrics
            'holes_created': info.get('holes_created', 0),
            'max_height': info.get('max_height', 0),
            'avg_height': info.get('avg_height', 0),
            'total_bumpiness': info.get('total_bumpiness', 0),
            'wells_created': info.get('wells_created', 0),
            'coverage': info.get('coverage', 0),
            'max_combo': info.get('max_combo', 0),
            'unique_states': info.get('unique_states', 0),
            'state_revisits': info.get('state_revisits', 0),
        }
        
        # Calculate derived metrics
        if episode_data['total_pieces'] > 0:
            episode_data['lines_per_piece'] = episode_data['lines_cleared'] / episode_data['total_pieces']
            episode_data['score_per_piece'] = episode_data['score'] / episode_data['total_pieces']
        else:
            episode_data['lines_per_piece'] = 0
            episode_data['score_per_piece'] = 0
        
        if episode_data['actions_taken'] > 0:
            episode_data['invalid_action_rate'] = episode_data['invalid_actions'] / episode_data['actions_taken']
        else:
            episode_data['invalid_action_rate'] = 0
        
        # Tetris rate (4-line clears / total line clearing events)
        total_clear_events = (episode_data['single_clears'] + episode_data['double_clears'] + 
                             episode_data['triple_clears'] + episode_data['tetris_clears'])
        episode_data['tetris_rate'] = (episode_data['tetris_clears'] / max(1, total_clear_events))
        
        self.episode_data.append(episode_data)
        self.total_episodes += 1
        self.running_scores.append(episode_data['score'])
        self.running_lines.append(episode_data['lines_cleared'])
        
        # Log to console every 100 episodes
        if self.total_episodes % 100 == 0:
            recent_scores = self.running_scores[-100:]
            recent_lines = self.running_lines[-100:]
            
            print(f"\n{'='*60}")
            print(f"Episode {self.total_episodes} | Timestep {self.num_timesteps}")
            print(f"Avg Score (last 100): {np.mean(recent_scores):.1f}")
            print(f"Avg Lines (last 100): {np.mean(recent_lines):.1f}")
            print(f"Max Score (last 100): {np.max(recent_scores):.0f}")
            print(f"Max Lines (last 100): {np.max(recent_lines):.0f}")
            print(f"{'='*60}\n")
    
    def save_metrics(self):
        """Save all collected metrics to files"""
        # Save episode data
        df = pd.DataFrame(self.episode_data)
        csv_path = self.save_path / 'episode_metrics.csv'
        df.to_csv(csv_path, index=False)
        
        # Calculate aggregate metrics
        aggregate = {
            'total_episodes': int(self.total_episodes),
            'avg_score': float(df['score'].mean()),
            'max_score': float(df['score'].max()),
            'avg_lines': float(df['lines_cleared'].mean()),
            'max_lines': float(df['lines_cleared'].max()),
            'avg_survival_time': float(df['survival_time'].mean()),
            'avg_pieces_placed': float(df['total_pieces'].mean()),
            'avg_lines_per_piece': float(df['lines_per_piece'].mean()),
            'avg_tetris_rate': float(df['tetris_rate'].mean()),
            'avg_invalid_action_rate': float(df['invalid_action_rate'].mean()),
            
            # Action distribution
            'action_distribution': {
                'rotations': int(df['rotations'].sum()),
                'left_moves': int(df['left_moves'].sum()),
                'right_moves': int(df['right_moves'].sum()),
                'soft_drops': int(df['soft_drops'].sum()),
                'hard_drops': int(df['hard_drops'].sum()),
                'no_ops': int(df['no_ops'].sum()),
            },
            
            # Piece usage
            'piece_usage': {
                'I': int(df['piece_I'].sum()),
                'O': int(df['piece_O'].sum()),
                'T': int(df['piece_T'].sum()),
                'L': int(df['piece_L'].sum()),
                'J': int(df['piece_J'].sum()),
                'S': int(df['piece_S'].sum()),
                'Z': int(df['piece_Z'].sum()),
            },
            
            # Line clearing breakdown
            'line_clears': {
                'singles': int(df['single_clears'].sum()),
                'doubles': int(df['double_clears'].sum()),
                'triples': int(df['triple_clears'].sum()),
                'tetrises': int(df['tetris_clears'].sum()),
            }
        }
        
        json_path = self.save_path / 'aggregate_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        print(f"\n✓ Metrics saved to {self.save_path}")
        
        return df, aggregate


def make_tetris_env(rank: int = 0, seed: int = 42, log_dir: Optional[str] = None, play_style: str = 'balanced'):
    """Create a wrapped Tetris environment"""
    def _init():
        try:
            # Add the env directory to the Python path
            env_dir = Path(__file__).parent.parent / 'env'
            sys.path.insert(0, str(env_dir))
            from tetris_env import TetrisEnv
        except ImportError:
            print("Error: Cannot import TetrisEnv. Make sure tetris_env.py is in PYTHONPATH.")
            print(f"Looked in: {env_dir}")
            sys.exit(1)
        
        # Create the environment with the seed in the reset kwargs
        env = TetrisEnv(
            render_mode=None,
            max_steps=10000,
            gravity_interval=10,
            seed=seed + rank,
            play_style=play_style
        )
        
        # Wrap with Monitor if log directory is provided
        if log_dir is not None:
            env = Monitor(env, log_dir)
        
        return env
    
    return _init


def plot_training_metrics(metrics_df: pd.DataFrame, save_path: Path, window: int = 100):
    """Generate comprehensive training plots for Tetris"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Tetris Training Metrics', fontsize=16)
    
    # 1. Score over time
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_df['score_rolling'] = metrics_df['score'].rolling(window=window, min_periods=1).mean()
    ax1.plot(metrics_df['episode_num'], metrics_df['score_rolling'], linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Score (Rolling {window} episodes)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Lines cleared over time
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_df['lines_rolling'] = metrics_df['lines_cleared'].rolling(window=window, min_periods=1).mean()
    ax2.plot(metrics_df['episode_num'], metrics_df['lines_rolling'], linewidth=2, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Lines Cleared')
    ax2.set_title(f'Lines Cleared (Rolling {window} episodes)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Survival time over time
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_df['survival_rolling'] = metrics_df['survival_time'].rolling(window=window, min_periods=1).mean()
    ax3.plot(metrics_df['episode_num'], metrics_df['survival_rolling'], linewidth=2, color='orange')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title(f'Survival Time (Rolling {window} episodes)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Lines per piece efficiency
    ax4 = fig.add_subplot(gs[0, 3])
    metrics_df['lpp_rolling'] = metrics_df['lines_per_piece'].rolling(window=window, min_periods=1).mean()
    ax4.plot(metrics_df['episode_num'], metrics_df['lpp_rolling'], linewidth=2, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Lines per Piece')
    ax4.set_title(f'Efficiency (Rolling {window} episodes)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Score distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(metrics_df['score'], bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(x=metrics_df['score'].mean(), color='r', linestyle='--', 
                label=f"Mean: {metrics_df['score'].mean():.0f}")
    ax5.set_xlabel('Score')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Score Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Line clearing breakdown
    ax6 = fig.add_subplot(gs[1, 1])
    clear_types = ['single_clears', 'double_clears', 'triple_clears', 'tetris_clears']
    clear_sums = [metrics_df[ct].sum() for ct in clear_types]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    ax6.bar(range(len(clear_types)), clear_sums, color=colors, edgecolor='black', alpha=0.7)
    ax6.set_xticks(range(len(clear_types)))
    ax6.set_xticklabels(['Single', 'Double', 'Triple', 'Tetris'])
    ax6.set_ylabel('Count')
    ax6.set_title('Line Clear Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Action distribution
    ax7 = fig.add_subplot(gs[1, 2])
    action_types = ['left_moves', 'right_moves', 'rotations', 'soft_drops', 'hard_drops', 'no_ops']
    action_sums = [metrics_df[at].sum() for at in action_types]
    ax7.bar(range(len(action_types)), action_sums, edgecolor='black', alpha=0.7)
    ax7.set_xticks(range(len(action_types)))
    ax7.set_xticklabels(['Left', 'Right', 'Rotate', 'Soft\nDrop', 'Hard\nDrop', 'No-op'], fontsize=8)
    ax7.set_ylabel('Count')
    ax7.set_title('Action Distribution')
    ax7.grid(True, alpha=0.3, axis='y')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)
    
    # 8. Piece usage distribution
    ax8 = fig.add_subplot(gs[1, 3])
    piece_types = ['piece_I', 'piece_O', 'piece_T', 'piece_L', 'piece_J', 'piece_S', 'piece_Z']
    piece_sums = [metrics_df[pt].sum() for pt in piece_types]
    piece_colors = ['cyan', 'yellow', 'purple', 'orange', 'blue', 'green', 'red']
    ax8.bar(range(len(piece_types)), piece_sums, color=piece_colors, edgecolor='black', alpha=0.7)
    ax8.set_xticks(range(len(piece_types)))
    ax8.set_xticklabels(['I', 'O', 'T', 'L', 'J', 'S', 'Z'])
    ax8.set_ylabel('Count')
    ax8.set_title('Piece Usage Distribution')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Invalid actions rate
    ax9 = fig.add_subplot(gs[2, 0])
    metrics_df['invalid_rolling'] = metrics_df['invalid_action_rate'].rolling(window=window, min_periods=1).mean()
    ax9.plot(metrics_df['episode_num'], metrics_df['invalid_rolling'], linewidth=2, color='red')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Invalid Action Rate')
    ax9.set_title(f'Invalid Actions (Rolling {window} episodes)')
    ax9.grid(True, alpha=0.3)
    
    # 10. Max height over time
    ax10 = fig.add_subplot(gs[2, 1])
    metrics_df['height_rolling'] = metrics_df['max_height'].rolling(window=window, min_periods=1).mean()
    ax10.plot(metrics_df['episode_num'], metrics_df['height_rolling'], linewidth=2, color='brown')
    ax10.set_xlabel('Episode')
    ax10.set_ylabel('Max Height')
    ax10.set_title(f'Max Height (Rolling {window} episodes)')
    ax10.grid(True, alpha=0.3)
    
    # 11. Tetris rate over time
    ax11 = fig.add_subplot(gs[2, 2])
    metrics_df['tetris_rate_rolling'] = metrics_df['tetris_rate'].rolling(window=window, min_periods=1).mean()
    ax11.plot(metrics_df['episode_num'], metrics_df['tetris_rate_rolling'], linewidth=2, color='darkgreen')
    ax11.set_xlabel('Episode')
    ax11.set_ylabel('Tetris Rate')
    ax11.set_title(f'Tetris Rate (Rolling {window} episodes)')
    ax11.grid(True, alpha=0.3)
    
    # 12. Max combo over time
    ax12 = fig.add_subplot(gs[2, 3])
    metrics_df['combo_rolling'] = metrics_df['max_combo'].rolling(window=window, min_periods=1).mean()
    ax12.plot(metrics_df['episode_num'], metrics_df['combo_rolling'], linewidth=2, color='magenta')
    ax12.set_xlabel('Episode')
    ax12.set_ylabel('Max Combo')
    ax12.set_title(f'Max Combo (Rolling {window} episodes)')
    ax12.grid(True, alpha=0.3)
    
    plt.savefig(save_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {save_path / 'training_metrics.png'}")
    plt.close()


def train(
    algo: str = 'ppo',
    play_style: str = 'balanced',
    timesteps: int = 100000,
    seed: int = 42,
    n_envs: int = 4,
    save_freq: int = 50000,
    eval_freq: int = 10000,
    output_dir: str = 'models',
    logs_dir: str = 'logs',
    experiment_name: Optional[str] = None,
    config: Optional[Dict] = None
):
    """
    Main training function for Tetris DRL agents.
    """
    # Set seeds
    np.random.seed(seed)
    
    # Separate environment config from algorithm config
    env_config = {}
    algo_config = {}
    if config:
        env_config = config.get('persona_weights', {})
        algo_config = config.get('algo_config', {})
    
    # Create experiment directories
    exp_name = experiment_name or f"{algo}_{play_style}_seed{seed}"
    
    # Set up models directory
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logs directory
    log_dir = Path(logs_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_data = {
        'algo': algo,
        'timesteps': timesteps,
        'seed': seed,
        'n_envs': n_envs,
        'save_freq': save_freq,
        'eval_freq': eval_freq,
        'config': config or {}
    }
    
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"Starting Tetris Training: {exp_name}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Play Style: {play_style}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Environments: {n_envs}")
    print(f"Models output: {model_dir}")
    print(f"Logs output: {log_dir}")
    print(f"{'='*60}\n")
    
    # Create vectorized environments
    env = DummyVecEnv([
        make_tetris_env(play_style=play_style, rank=i, seed=seed, log_dir=str(log_dir))
        for i in range(n_envs)
    ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_tetris_env(play_style=play_style, rank=0, seed=seed + 10000)])
    
    # Create model name with play style
    model_name = f"{algo}_{play_style}"
    
    # Create model
    algo = algo.lower()
    common_params = {
        'env': env,
        'seed': seed,
        'verbose': 1,
    }
    
    if algo == 'ppo':
        model = PPO(
            policy='MlpPolicy',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common_params,
            **algo_config
        )
    elif algo == 'a2c':
        model = A2C(
            policy='MlpPolicy',
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            **common_params,
            **(config or {})
        )
    elif algo == 'dqn':
        # DQN works best with single environment
        env.close()
        env = DummyVecEnv([make_tetris_env(rank=0, seed=seed, log_dir=str(log_dir))])
        model = DQN(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            env=env,
            seed=seed,
            verbose=1,
            **(config or {})
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Configure logger
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    
    # Create callbacks
    metrics_callback = TetrisMetricsCallback(save_path=log_dir / 'metrics')
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / f'{exp_name}_best'),
        log_path=str(log_dir / 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"\nTraining {algo.upper()} agent...\n")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[metrics_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save final model with play style
    model_path = model_dir / f'{model_name}_seed{seed}.zip'
    model.save(model_path)
    print(f"\n✓ Final model saved to {model_path}")
    
    # Save and plot metrics
    metrics_df, aggregate = metrics_callback.save_metrics()
    plot_training_metrics(metrics_df, log_dir / 'metrics')
    
    # Save final summary
    summary = {
        **config_data,
        'training_time_seconds': training_time,
        'final_metrics': aggregate
    }
    
    with open(log_dir / 'final_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {training_time:.2f}s ({training_time/60:.1f} minutes)")
    print(f"Avg Score: {aggregate['avg_score']:.1f}")
    print(f"Max Score: {aggregate['max_score']:.0f}")
    print(f"Avg Lines: {aggregate['avg_lines']:.1f}")
    print(f"Max Lines: {aggregate['max_lines']:.0f}")
    print(f"Model saved to: {model_path}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")
    
    env.close()
    eval_env.close()
    
    return model, metrics_df, aggregate


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train DRL agents for Tetris game testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'a2c', 'dqn'],
                       help='RL algorithm to use')
    parser.add_argument('--play-style', type=str, default='balanced',
                       choices=['balanced', 'aggressive', 'conservative', 'speedrun'],
                       help='Play style for the reward system')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory for saving trained models')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory for saving training logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    # Frequency parameters
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Model checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Run training
    train(
        algo=args.algo,
        play_style=args.play_style,
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
        logs_dir=args.logs_dir,
        experiment_name=args.experiment_name,
        config=config
    )


if __name__ == '__main__':
    main()