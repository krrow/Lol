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

# Import the Wordle environment
# from envs.wordle_env import WordleEnv  # Adjust import path as needed


class MetricsCallback(BaseCallback):
    """
    Custom callback for collecting detailed metrics during training.
    Logs per-episode metrics and aggregates for analysis.
    """
    
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Episode metrics
        self.episode_data = []
        self.current_episode = {
            'rewards': [],
            'guesses': [],
            'letters_tried': set(),
            'repeated_guesses': 0
        }
        
        # Aggregated metrics
        self.total_wins = 0
        self.total_episodes = 0
        self.win_rates = []
        self.guess_distributions = []
        
    def _on_step(self) -> bool:
        """Called at each environment step"""
        # Check if episode ended
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            infos = self.locals['infos']
            
            for i, done in enumerate(dones):
                if done and 'episode' in infos[i]:
                    self._log_episode(infos[i])
        
        return True
    
    def _log_episode(self, info: Dict):
        """Log metrics for a completed episode"""
        episode_info = info.get('episode', {})
        
        episode_data = {
            'timestep': self.num_timesteps,
            'episode_num': self.total_episodes,
            'reward': episode_info.get('r', 0),
            'length': episode_info.get('l', 0),
            'won': episode_info.get('won', False),
            'target_word': info.get('target_word', ''),
            'guess_count': info.get('guess_count', 0),
            'possible_words_count': info.get('possible_words_count', 0),
        }
        
        self.episode_data.append(episode_data)
        self.total_episodes += 1
        
        if episode_data['won']:
            self.total_wins += 1
            self.guess_distributions.append(episode_data['guess_count'])
        
        # Calculate rolling win rate
        if self.total_episodes > 0:
            win_rate = self.total_wins / self.total_episodes
            self.win_rates.append(win_rate)
        
        # Log to console every 100 episodes
        if self.total_episodes % 100 == 0:
            recent_wins = sum(1 for ep in self.episode_data[-100:] if ep['won'])
            recent_win_rate = recent_wins / min(100, len(self.episode_data))
            avg_guesses = np.mean([ep['guess_count'] for ep in self.episode_data[-100:]])
            
            print(f"\n{'='*60}")
            print(f"Episode {self.total_episodes} | Timestep {self.num_timesteps}")
            print(f"Recent Win Rate (last 100): {recent_win_rate:.2%}")
            print(f"Overall Win Rate: {self.total_wins / self.total_episodes:.2%}")
            print(f"Avg Guesses (last 100): {avg_guesses:.2f}")
            print(f"{'='*60}\n")
    
    def save_metrics(self):
        """Save all collected metrics to files"""
        # Save episode data
        df = pd.DataFrame(self.episode_data)
        csv_path = self.save_path / 'episode_metrics.csv'
        df.to_csv(csv_path, index=False)
        
        # Save aggregate metrics
        aggregate = {
            'total_episodes': self.total_episodes,
            'total_wins': self.total_wins,
            'overall_win_rate': self.total_wins / max(1, self.total_episodes),
            'avg_guesses_when_won': np.mean(self.guess_distributions) if self.guess_distributions else 0,
            'guess_distribution': {
                f'{i}_guesses': self.guess_distributions.count(i) 
                for i in range(1, 7)
            }
        }
        
        json_path = self.save_path / 'aggregate_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        print(f"\n✓ Metrics saved to {self.save_path}")
        
        return df, aggregate


class WordleMaskedEnv(gym.Wrapper):
    """
    Wrapper to handle action masking for invalid words.
    Converts masked discrete actions to valid discrete actions.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.use_masking = env.use_action_masking
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add action mask to observation if needed
        if self.use_masking and 'action_mask' in info:
            obs['action_mask'] = info['action_mask']
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.use_masking and 'action_mask' in info:
            obs['action_mask'] = info['action_mask']
        
        return obs, info


def make_env(
    env_class,
    persona: str,
    seed: int,
    rank: int = 0,
    use_action_masking: bool = True,
    log_dir: Optional[str] = None
):
    """
    Create a wrapped Wordle environment.
    """
    def _init():
        env = env_class(
            persona=persona,
            use_action_masking=use_action_masking,
            max_guesses=6
        )
        env = WordleMaskedEnv(env)
        
        if log_dir is not None:
            env = Monitor(env, log_dir)
        
        env.reset(seed=seed + rank)
        return env
    
    return _init


def create_model(
    algo: str,
    env,
    seed: int,
    learning_rate: float = 3e-4,
    **kwargs
) -> Tuple:
    """
    Create a DRL model based on algorithm name.
    
    Args:
        algo: Algorithm name ('ppo', 'a2c', 'dqn')
        env: Training environment
        seed: Random seed
        learning_rate: Learning rate
        **kwargs: Additional algorithm-specific parameters
    
    Returns:
        Tuple of (model, model_class_name)
    """
    algo = algo.lower()
    
    # Common parameters
    common_params = {
        'env': env,
        'seed': seed,
        'verbose': 1,
        'learning_rate': learning_rate,
    }
    
    if algo == 'ppo':
        model = PPO(
            policy='MultiInputPolicy',
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            **common_params,
            **kwargs
        )
    elif algo == 'a2c':
        model = A2C(
            policy='MultiInputPolicy',
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            **common_params,
            **kwargs
        )
    elif algo == 'dqn':
        model = DQN(
            policy='MultiInputPolicy',
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            **common_params,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return model, algo.upper()


def plot_training_metrics(
    metrics_df: pd.DataFrame,
    save_path: Path,
    window: int = 100
):
    """
    Generate training plots from metrics dataframe.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Wordle Training Metrics', fontsize=16)
    
    # 1. Win rate over time
    ax1 = axes[0, 0]
    metrics_df['win_rate_rolling'] = metrics_df['won'].rolling(window=window).mean()
    ax1.plot(metrics_df['episode_num'], metrics_df['win_rate_rolling'])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.set_title(f'Win Rate (Rolling {window} episodes)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode rewards
    ax2 = axes[0, 1]
    metrics_df['reward_rolling'] = metrics_df['reward'].rolling(window=window).mean()
    ax2.plot(metrics_df['episode_num'], metrics_df['reward_rolling'])
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Episode Reward (Rolling {window} episodes)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Guess count distribution (for wins only)
    ax3 = axes[1, 0]
    wins_df = metrics_df[metrics_df['won'] == True]
    if len(wins_df) > 0:
        ax3.hist(wins_df['guess_count'], bins=range(1, 8), edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Number of Guesses')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Guess Distribution (Wins Only)')
        ax3.set_xticks(range(1, 7))
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Average guesses over time (wins only)
    ax4 = axes[1, 1]
    if len(wins_df) > 0:
        wins_df['guess_rolling'] = wins_df['guess_count'].rolling(window=window, min_periods=1).mean()
        ax4.plot(wins_df['episode_num'], wins_df['guess_rolling'])
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Avg Guesses')
        ax4.set_title(f'Average Guesses to Win (Rolling {window} episodes)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {save_path / 'training_metrics.png'}")
    plt.close()


def train(
    algo: str = 'ppo',
    persona: str = 'explorer',
    timesteps: int = 100000,
    seed: int = 42,
    n_envs: int = 4,
    save_freq: int = 10000,
    eval_freq: int = 5000,
    output_dir: str = 'models',
    logs_dir: str = 'logs',
    experiment_name: Optional[str] = None,
    config: Optional[Dict] = None
):
    """
    Main training function.
    
    Args:
        algo: Algorithm to use ('ppo', 'a2c', 'dqn')
        persona: Reward persona ('explorer', 'speedrunner', 'validator', 'survivor')
        timesteps: Total training timesteps
        seed: Random seed for reproducibility
        n_envs: Number of parallel environments
        save_freq: Frequency to save model checkpoints
        eval_freq: Frequency to run evaluation
        output_dir: Directory for model outputs (default: 'models')
        logs_dir: Directory for training logs (default: 'logs')
        experiment_name: Custom experiment name
        config: Optional config dict with additional parameters
    """
    # Set seeds
    np.random.seed(seed)
    
    # Create experiment name
    exp_name = experiment_name or f"{algo}_{persona}_seed{seed}"
    
    # Create model and logs directories
    model_dir = Path(output_dir)
    log_dir = Path(logs_dir) / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model path
    model_path = model_dir / f"{exp_name}.zip"
    
    # Save configuration to logs directory
    config_data = {
        'algo': algo,
        'persona': persona,
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
    print(f"Starting Training: {exp_name}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Persona: {persona}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Model Output: {model_path}")
    print(f"Logs Output: {log_dir}")
    print(f"{'='*60}\n")
    
    # Import WordleEnv from env directory
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
        from wordle_env import WordleEnv
    except ImportError:
        print("Error: Cannot import WordleEnv. Make sure wordle_env.py is in the env directory.")
        sys.exit(1)
    
    # Create vectorized environments
    env = DummyVecEnv([
        make_env(WordleEnv, persona, seed, i, log_dir=str(log_dir / 'monitor'))
        for i in range(n_envs)
    ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(WordleEnv, persona, seed + 1000, 0)
    ])
    
    # Create model
    model, model_name = create_model(
        algo=algo,
        env=env,
        seed=seed,
        **(config or {})
    )
    
    # Configure logger
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    
    # Create callbacks
    metrics_callback = MetricsCallback(save_path=log_dir / 'metrics')
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / f"{exp_name}_best"),
        log_path=str(log_dir / 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=50,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"\nTraining {model_name} agent...\n")
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
    
    # Save final model
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
    
    with open(log_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {training_time:.2f}s")
    print(f"Final Win Rate: {aggregate['overall_win_rate']:.2%}")
    print(f"Avg Guesses (Wins): {aggregate['avg_guesses_when_won']:.2f}")
    print(f"Model saved to: {model_path}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")
    
    env.close()
    eval_env.close()
    
    return model, metrics_df, aggregate


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train DRL agents for Wordle game testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'a2c', 'dqn'],
                       help='RL algorithm to use')
    parser.add_argument('--persona', type=str, default='explorer',
                       choices=['explorer', 'speedrunner', 'validator', 'survivor'],
                       help='Reward persona')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory for model outputs')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory for training logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    # Frequency parameters
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Model checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=5000,
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
        persona=args.persona,
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
