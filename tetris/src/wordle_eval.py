import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv


def load_model(model_path: str):
    """Load a trained model"""
    model_path = Path(model_path)
    
    # Try to determine algorithm from parent directory
    if 'ppo' in str(model_path).lower():
        return PPO.load(model_path)
    elif 'a2c' in str(model_path).lower():
        return A2C.load(model_path)
    elif 'dqn' in str(model_path).lower():
        return DQN.load(model_path)
    else:
        # Try each algorithm
        for algo_class in [PPO, A2C, DQN]:
            try:
                return algo_class.load(model_path)
            except:
                continue
        raise ValueError(f"Could not load model from {model_path}")


def evaluate_model(
    model,
    env,
    n_episodes: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained model and collect detailed metrics.
    
    Returns:
        Dictionary with evaluation metrics and episode details
    """
    episodes_data = []
    word_success = defaultdict(list)  # Track success per word
    guess_sequences = []  # Track guess patterns
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_guesses = []
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            # Log guess if available
            if 'last_guess' in info[0]:
                episode_guesses.append(info[0]['last_guess'])
        
        # Extract episode info
        episode_info = {
            'episode': episode,
            'reward': episode_reward,
            'won': info[0].get('won', False),
            'guess_count': info[0].get('guess_count', 0),
            'target_word': info[0].get('target_word', ''),
            'guesses': episode_guesses,
            'unique_guesses': len(set(episode_guesses)),
            'repeated_guesses': len(episode_guesses) - len(set(episode_guesses))
        }
        
        episodes_data.append(episode_info)
        
        # Track per-word success
        target = episode_info['target_word']
        word_success[target].append(episode_info['won'])
        
        # Track guess sequences
        if episode_info['won']:
            guess_sequences.append(tuple(episode_guesses))
        
        if verbose and (episode + 1) % 100 == 0:
            recent_wins = sum(ep['won'] for ep in episodes_data[-100:])
            print(f"Episodes: {episode + 1}/{n_episodes} | Recent Win Rate: {recent_wins}%")
    
    # Calculate aggregate metrics
    df = pd.DataFrame(episodes_data)
    
    wins = df[df['won'] == True]
    losses = df[df['won'] == False]
    
    metrics = {
        'total_episodes': n_episodes,
        'total_wins': len(wins),
        'total_losses': len(losses),
        'win_rate': len(wins) / n_episodes,
        'avg_reward': df['reward'].mean(),
        'avg_guesses': df['guess_count'].mean(),
        'avg_guesses_when_won': wins['guess_count'].mean() if len(wins) > 0 else 0,
        'avg_guesses_when_lost': losses['guess_count'].mean() if len(losses) > 0 else 0,
        'avg_unique_guesses': df['unique_guesses'].mean(),
        'avg_repeated_guesses': df['repeated_guesses'].mean(),
        'guess_distribution': {
            i: len(wins[wins['guess_count'] == i]) 
            for i in range(1, 7)
        },
        'hardest_words': [],
        'easiest_words': [],
        'most_common_winning_sequences': []
    }
    
    # Find hardest/easiest words (with sufficient data)
    word_win_rates = {
        word: (sum(successes) / len(successes), len(successes))
        for word, successes in word_success.items()
        if len(successes) >= 5  # At least 5 attempts
    }
    
    if word_win_rates:
        sorted_words = sorted(word_win_rates.items(), key=lambda x: x[1][0])
        metrics['hardest_words'] = [
            {'word': w, 'win_rate': wr, 'attempts': n}
            for w, (wr, n) in sorted_words[:10]
        ]
        metrics['easiest_words'] = [
            {'word': w, 'win_rate': wr, 'attempts': n}
            for w, (wr, n) in sorted_words[-10:]
        ]
    
    # Most common winning sequences
    if guess_sequences:
        sequence_counts = Counter(guess_sequences)
        metrics['most_common_winning_sequences'] = [
            {'sequence': list(seq), 'count': count}
            for seq, count in sequence_counts.most_common(10)
        ]
    
    return metrics, df


def plot_evaluation_results(
    df: pd.DataFrame,
    metrics: Dict,
    save_path: Path,
    title_prefix: str = ""
):
    """Generate comprehensive evaluation plots"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    if title_prefix:
        fig.suptitle(f'{title_prefix} - Evaluation Results', fontsize=16, y=0.995)
    
    # 1. Win Rate over episodes
    ax1 = fig.add_subplot(gs[0, 0])
    df['win_rate_rolling'] = df['won'].rolling(window=100, min_periods=1).mean()
    ax1.plot(df['episode'], df['win_rate_rolling'])
    ax1.axhline(y=metrics['win_rate'], color='r', linestyle='--', alpha=0.5, label='Overall')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate (Rolling 100 episodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['reward'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=metrics['avg_reward'], color='r', linestyle='--', label=f"Mean: {metrics['avg_reward']:.2f}")
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Guess distribution (wins only)
    ax3 = fig.add_subplot(gs[0, 2])
    wins = df[df['won'] == True]
    if len(wins) > 0:
        guess_counts = [metrics['guess_distribution'][i] for i in range(1, 7)]
        ax3.bar(range(1, 7), guess_counts, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Number of Guesses')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Guess Distribution (Wins Only, n={len(wins)})')
        ax3.set_xticks(range(1, 7))
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Win rate by guess count
    ax4 = fig.add_subplot(gs[1, 0])
    guess_win_rates = df.groupby('guess_count')['won'].mean()
    ax4.bar(guess_win_rates.index, guess_win_rates.values, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Guess Count')
    ax4.set_ylabel('Win Rate')
    ax4.set_title('Win Rate by Guess Count')
    ax4.set_xticks(range(1, 7))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Unique vs repeated guesses
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df['unique_guesses'], df['repeated_guesses'], alpha=0.3, s=10)
    ax5.set_xlabel('Unique Guesses')
    ax5.set_ylabel('Repeated Guesses')
    ax5.set_title('Guess Diversity')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative wins
    ax6 = fig.add_subplot(gs[1, 2])
    df['cumulative_wins'] = df['won'].cumsum()
    ax6.plot(df['episode'], df['cumulative_wins'])
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Cumulative Wins')
    ax6.set_title('Cumulative Wins Over Time')
    ax6.grid(True, alpha=0.3)
    
    # 7. Win/Loss pie chart
    ax7 = fig.add_subplot(gs[2, 0])
    sizes = [metrics['total_wins'], metrics['total_losses']]
    labels = [f"Wins ({metrics['total_wins']})", f"Losses ({metrics['total_losses']})"]
    colors = ['#2ecc71', '#e74c3c']
    ax7.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax7.set_title(f"Overall Win Rate: {metrics['win_rate']:.1%}")
    
    # 8. Top 10 hardest words
    ax8 = fig.add_subplot(gs[2, 1])
    if metrics['hardest_words']:
        words = [w['word'] for w in metrics['hardest_words'][:10]]
        win_rates = [w['win_rate'] for w in metrics['hardest_words'][:10]]
        y_pos = np.arange(len(words))
        ax8.barh(y_pos, win_rates, alpha=0.7)
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(words, fontsize=8)
        ax8.set_xlabel('Win Rate')
        ax8.set_title('Top 10 Hardest Words')
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. Reward over time
    ax9 = fig.add_subplot(gs[2, 2])
    df['reward_rolling'] = df['reward'].rolling(window=100, min_periods=1).mean()
    ax9.plot(df['episode'], df['reward_rolling'])
    ax9.axhline(y=metrics['avg_reward'], color='r', linestyle='--', alpha=0.5, label='Overall')
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Reward')
    ax9.set_title('Episode Reward (Rolling 100 episodes)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.savefig(save_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation plots saved to {save_path / 'evaluation_results.png'}")
    plt.close()


def compare_models(results: List[Dict], save_path: Path):
    """Generate comparison plots for multiple models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    model_names = [r['name'] for r in results]
    
    # 1. Win rates
    ax1 = axes[0, 0]
    win_rates = [r['metrics']['win_rate'] for r in results]
    ax1.bar(model_names, win_rates, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate Comparison')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Average guesses when won
    ax2 = axes[0, 1]
    avg_guesses = [r['metrics']['avg_guesses_when_won'] for r in results]
    ax2.bar(model_names, avg_guesses, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Avg Guesses')
    ax2.set_title('Average Guesses (Wins Only)')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Average reward
    ax3 = axes[0, 2]
    avg_rewards = [r['metrics']['avg_reward'] for r in results]
    ax3.bar(model_names, avg_rewards, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Avg Reward')
    ax3.set_title('Average Episode Reward')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Guess distribution comparison
    ax4 = axes[1, 0]
    x = np.arange(1, 7)
    width = 0.8 / len(results)
    for i, result in enumerate(results):
        counts = [result['metrics']['guess_distribution'][j] for j in range(1, 7)]
        offset = (i - len(results)/2 + 0.5) * width
        ax4.bar(x + offset, counts, width, label=result['name'], alpha=0.7)
    ax4.set_xlabel('Number of Guesses')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Guess Distribution Comparison (Wins)')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Repeated guesses
    ax5 = axes[1, 1]
    repeated = [r['metrics']['avg_repeated_guesses'] for r in results]
    ax5.bar(model_names, repeated, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Avg Repeated Guesses')
    ax5.set_title('Average Repeated Guesses')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Summary table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = []
    for r in results:
        m = r['metrics']
        table_data.append([
            r['name'],
            f"{m['win_rate']:.2%}",
            f"{m['avg_guesses_when_won']:.2f}",
            f"{m['avg_reward']:.2f}"
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=['Model', 'Win Rate', 'Avg Guesses', 'Avg Reward'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to {save_path / 'model_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained Wordle DRL agents',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file(s). Use wildcards for multiple models.')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--persona', type=str, default='explorer',
                       choices=['explorer', 'speedrunner', 'validator', 'survivor'],
                       help='Environment persona (should match training)')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                       help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models (use wildcards in --model)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes (slows down evaluation)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Import WordleEnv
    try:
        from wordle_env import WordleEnv
    except ImportError:
        print("Error: Cannot import WordleEnv. Make sure wordle_env.py is in PYTHONPATH.")
        sys.exit(1)
    
    # Find model files
    from glob import glob
    model_paths = glob(args.model)
    
    if not model_paths:
        print(f"Error: No model files found matching '{args.model}'")
        sys.exit(1)
    
    print(f"\nFound {len(model_paths)} model(s) to evaluate")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for model_path in model_paths:
        model_path = Path(model_path)
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path.name}")
        print(f"{'='*60}\n")
        
        # Load model
        try:
            model = load_model(model_path)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            continue
        
        # Create environment
        env = DummyVecEnv([
            lambda: WordleEnv(persona=args.persona, use_action_masking=True)
        ])
        
        # Evaluate
        print(f"\nRunning {args.episodes} evaluation episodes...")
        metrics, df = evaluate_model(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=args.deterministic,
            render=args.render,
            verbose=True
        )
        
        # Create result directory
        model_name = model_path.parent.parent.name if 'best_model' in str(model_path) else model_path.parent.name
        result_dir = output_dir / model_name
        result_dir.mkdir(exist_ok=True)
        
        # Save results
        df.to_csv(result_dir / 'episodes.csv', index=False)
        
        with open(result_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot results
        plot_evaluation_results(df, metrics, result_dir, title_prefix=model_name)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Summary: {model_name}")
        print(f"{'='*60}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Wins: {metrics['total_wins']}/{metrics['total_episodes']}")
        print(f"Avg Guesses (Overall): {metrics['avg_guesses']:.2f}")
        print(f"Avg Guesses (Wins): {metrics['avg_guesses_when_won']:.2f}")
        print(f"Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"Avg Repeated Guesses: {metrics['avg_repeated_guesses']:.2f}")
        print(f"\nGuess Distribution (Wins):")
        for i in range(1, 7):
            count = metrics['guess_distribution'][i]
            pct = count / max(1, metrics['total_wins']) * 100
            print(f"  {i} guesses: {count} ({pct:.1f}%)")
        
        if metrics['hardest_words']:
            print(f"\nTop 5 Hardest Words:")
            for w in metrics['hardest_words'][:5]:
                print(f"  {w['word']}: {w['win_rate']:.1%} ({w['attempts']} attempts)")
        
        print(f"\nResults saved to: {result_dir}")
        print(f"{'='*60}\n")
        
        results.append({
            'name': model_name,
            'path': str(model_path),
            'metrics': metrics,
            'df': df
        })
        
        env.close()
    
    # Generate comparison plots if multiple models
    if len(results) > 1 and args.compare:
        print(f"\nGenerating comparison plots...")
        compare_models(results, output_dir)
        
        # Save comparison table
        comparison_data = []
        for r in results:
            m = r['metrics']
            comparison_data.append({
                'Model': r['name'],
                'Win_Rate': m['win_rate'],
                'Total_Wins': m['total_wins'],
                'Avg_Guesses_Overall': m['avg_guesses'],
                'Avg_Guesses_Wins': m['avg_guesses_when_won'],
                'Avg_Reward': m['avg_reward'],
                'Avg_Repeated_Guesses': m['avg_repeated_guesses'],
                'Guess_1': m['guess_distribution'][1],
                'Guess_2': m['guess_distribution'][2],
                'Guess_3': m['guess_distribution'][3],
                'Guess_4': m['guess_distribution'][4],
                'Guess_5': m['guess_distribution'][5],
                'Guess_6': m['guess_distribution'][6],
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"✓ Comparison data saved to {output_dir / 'model_comparison.csv'}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()