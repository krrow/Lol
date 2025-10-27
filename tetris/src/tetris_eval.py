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
    n_episodes: int = 500,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained Tetris model and collect detailed metrics.
    
    Returns:
        Dictionary with evaluation metrics and episode details
    """
    episodes_data = []
    action_sequences = defaultdict(int)  # Track common action patterns
    piece_survival = defaultdict(list)  # Track survival by piece type
    height_progression = []  # Track how height changes
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        max_height_reached = 0
        done = False
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_actions.append(int(action[0]))
            step_count += 1
            
            # Track max height
            if 'max_height' in info[0]:
                max_height_reached = max(max_height_reached, info[0]['max_height'])
        
        # Extract episode info
        episode_info = {
            'episode': episode,
            'reward': episode_reward,
            'score': info[0].get('score', 0),
            'steps': info[0].get('steps', 0),
            'survival_time': info[0].get('survival_time', 0),
            'lines_cleared': info[0].get('lines_cleared', 0),
            'single_clears': info[0].get('single_clears', 0),
            'double_clears': info[0].get('double_clears', 0),
            'triple_clears': info[0].get('triple_clears', 0),
            'tetris_clears': info[0].get('tetris_clears', 0),
            'total_pieces': info[0].get('total_pieces', 0),
            'holes_created': info[0].get('holes_created', 0),
            'max_height': info[0].get('max_height', 0),
            'max_height_reached': max_height_reached,
            'avg_height': info[0].get('avg_height', 0),
            'total_bumpiness': info[0].get('total_bumpiness', 0),
            'wells_created': info[0].get('wells_created', 0),
            'coverage': info[0].get('coverage', 0),
            'max_combo': info[0].get('max_combo', 0),
            'rotations': info[0].get('rotations', 0),
            'left_moves': info[0].get('left_moves', 0),
            'right_moves': info[0].get('right_moves', 0),
            'soft_drops': info[0].get('soft_drops', 0),
            'hard_drops': info[0].get('hard_drops', 0),
            'no_ops': info[0].get('no_ops', 0),
            'invalid_actions': info[0].get('invalid_actions', 0),
            'unique_states': info[0].get('unique_states', 0),
            'state_revisits': info[0].get('state_revisits', 0),
        }
        
        # Calculate derived metrics
        if episode_info['total_pieces'] > 0:
            episode_info['lines_per_piece'] = episode_info['lines_cleared'] / episode_info['total_pieces']
            episode_info['score_per_piece'] = episode_info['score'] / episode_info['total_pieces']
        else:
            episode_info['lines_per_piece'] = 0
            episode_info['score_per_piece'] = 0
        
        total_actions = (episode_info['rotations'] + episode_info['left_moves'] + 
                        episode_info['right_moves'] + episode_info['soft_drops'] + 
                        episode_info['hard_drops'] + episode_info['no_ops'])
        
        if total_actions > 0:
            episode_info['invalid_action_rate'] = episode_info['invalid_actions'] / total_actions
        else:
            episode_info['invalid_action_rate'] = 0
        
        # Tetris rate
        total_clear_events = (episode_info['single_clears'] + episode_info['double_clears'] + 
                             episode_info['triple_clears'] + episode_info['tetris_clears'])
        episode_info['tetris_rate'] = (episode_info['tetris_clears'] / max(1, total_clear_events))
        
        # Track action patterns (3-action sequences)
        for i in range(len(episode_actions) - 2):
            pattern = tuple(episode_actions[i:i+3])
            action_sequences[pattern] += 1
        
        episodes_data.append(episode_info)
        height_progression.append(episode_info['max_height_reached'])
        
        if verbose and (episode + 1) % 50 == 0:
            recent = episodes_data[-50:]
            avg_score = np.mean([e['score'] for e in recent])
            avg_lines = np.mean([e['lines_cleared'] for e in recent])
            print(f"Episodes: {episode + 1}/{n_episodes} | Avg Score: {avg_score:.1f} | Avg Lines: {avg_lines:.1f}")
    
    # Calculate aggregate metrics
    df = pd.DataFrame(episodes_data)
    
    # Identify potential issues
    issues_detected = []
    
    # Issue 1: High invalid action rate
    if df['invalid_action_rate'].mean() > 0.2:
        issues_detected.append({
            'type': 'High Invalid Action Rate',
            'severity': 'Medium',
            'description': f"Agent attempts invalid actions {df['invalid_action_rate'].mean():.1%} of the time",
            'recommendation': 'Consider action masking or reward shaping to discourage invalid moves'
        })
    
    # Issue 2: Low line clearing efficiency
    if df['lines_per_piece'].mean() < 0.3:
        issues_detected.append({
            'type': 'Low Line Clearing Efficiency',
            'severity': 'High',
            'description': f"Only {df['lines_per_piece'].mean():.2f} lines cleared per piece",
            'recommendation': 'Agent may be stacking poorly; increase rewards for line clears'
        })
    
    # Issue 3: Excessive holes
    if df['holes_created'].mean() > 10:
        issues_detected.append({
            'type': 'Excessive Hole Creation',
            'severity': 'High',
            'description': f"Average of {df['holes_created'].mean():.1f} holes per game",
            'recommendation': 'Add penalties for creating holes in reward function'
        })
    
    # Issue 4: Poor height management
    if df['max_height'].mean() > 15:
        issues_detected.append({
            'type': 'Poor Height Management',
            'severity': 'High',
            'description': f"Average max height of {df['max_height'].mean():.1f} (danger zone)",
            'recommendation': 'Add stronger penalties for height increases'
        })
    
    # Issue 5: Low Tetris rate
    if df['tetris_rate'].mean() < 0.05:
        issues_detected.append({
            'type': 'Low Tetris (4-line) Rate',
            'severity': 'Low',
            'description': f"Only {df['tetris_rate'].mean():.1%} of clears are Tetrises",
            'recommendation': 'Consider bonus rewards for 4-line clears to encourage better play'
        })
    
    # Issue 6: Action repetition (stuck behavior)
    most_common_pattern = max(action_sequences.items(), key=lambda x: x[1])
    if most_common_pattern[1] > len(episodes_data) * 5:  # Same pattern >5 times per episode
        issues_detected.append({
            'type': 'Repetitive Action Pattern',
            'severity': 'Medium',
            'description': f"Agent frequently repeats action sequence {most_common_pattern[0]}",
            'recommendation': 'May indicate stuck behavior; check for policy collapse'
        })
    
    # Issue 7: Low survival time
    if df['survival_time'].mean() < 100:
        issues_detected.append({
            'type': 'Low Survival Time',
            'severity': 'Critical',
            'description': f"Games end quickly (avg {df['survival_time'].mean():.0f} steps)",
            'recommendation': 'Agent needs more training or better reward shaping'
        })
    
    metrics = {
        'total_episodes': n_episodes,
        'avg_score': df['score'].mean(),
        'max_score': df['score'].max(),
        'std_score': df['score'].std(),
        'avg_lines': df['lines_cleared'].mean(),
        'max_lines': df['lines_cleared'].max(),
        'std_lines': df['lines_cleared'].std(),
        'avg_survival_time': df['survival_time'].mean(),
        'max_survival_time': df['survival_time'].max(),
        'avg_pieces_placed': df['total_pieces'].mean(),
        'avg_lines_per_piece': df['lines_per_piece'].mean(),
        'avg_score_per_piece': df['score_per_piece'].mean(),
        'avg_tetris_rate': df['tetris_rate'].mean(),
        'avg_invalid_action_rate': df['invalid_action_rate'].mean(),
        'avg_holes': df['holes_created'].mean(),
        'avg_max_height': df['max_height'].mean(),
        'avg_bumpiness': df['total_bumpiness'].mean(),
        'avg_coverage': df['coverage'].mean(),
        'avg_max_combo': df['max_combo'].mean(),
        
        # Percentiles
        'score_p50': df['score'].quantile(0.5),
        'score_p90': df['score'].quantile(0.9),
        'score_p95': df['score'].quantile(0.95),
        'lines_p50': df['lines_cleared'].quantile(0.5),
        'lines_p90': df['lines_cleared'].quantile(0.9),
        'lines_p95': df['lines_cleared'].quantile(0.95),
        
        # Line clearing breakdown
        'total_singles': df['single_clears'].sum(),
        'total_doubles': df['double_clears'].sum(),
        'total_triples': df['triple_clears'].sum(),
        'total_tetrises': df['tetris_clears'].sum(),
        
        # Action distribution
        'total_rotations': df['rotations'].sum(),
        'total_left_moves': df['left_moves'].sum(),
        'total_right_moves': df['right_moves'].sum(),
        'total_soft_drops': df['soft_drops'].sum(),
        'total_hard_drops': df['hard_drops'].sum(),
        'total_no_ops': df['no_ops'].sum(),
        'total_invalid_actions': df['invalid_actions'].sum(),
        
        # Issues detected
        'issues_detected': issues_detected,
        
        # Most common action sequences
        'common_action_patterns': [
            {'pattern': list(k), 'count': v}
            for k, v in sorted(action_sequences.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    }
    
    return metrics, df


def plot_evaluation_results(
    df: pd.DataFrame,
    metrics: Dict,
    save_path: Path,
    title_prefix: str = ""
):
    """Generate comprehensive evaluation plots for Tetris"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    if title_prefix:
        fig.suptitle(f'{title_prefix} - Evaluation Results', fontsize=16, y=0.995)
    
    # 1. Score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['score'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax1.axvline(x=metrics['avg_score'], color='r', linestyle='--', label=f"Mean: {metrics['avg_score']:.1f}")
    ax1.axvline(x=metrics['score_p90'], color='g', linestyle='--', label=f"P90: {metrics['score_p90']:.1f}")
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Lines cleared distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['lines_cleared'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=metrics['avg_lines'], color='r', linestyle='--', label=f"Mean: {metrics['avg_lines']:.1f}")
    ax2.set_xlabel('Lines Cleared')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Lines Cleared Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Survival time distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['survival_time'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax3.axvline(x=metrics['avg_survival_time'], color='r', linestyle='--', 
                label=f"Mean: {metrics['avg_survival_time']:.0f}")
    ax3.set_xlabel('Survival Time (steps)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Survival Time Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Lines per piece efficiency
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(df['lines_per_piece'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(x=metrics['avg_lines_per_piece'], color='r', linestyle='--', 
                label=f"Mean: {metrics['avg_lines_per_piece']:.2f}")
    ax4.set_xlabel('Lines per Piece')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Efficiency Distribution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Line clear type breakdown
    ax5 = fig.add_subplot(gs[1, 0])
    clear_types = ['Singles', 'Doubles', 'Triples', 'Tetrises']
    clear_counts = [
        metrics['total_singles'],
        metrics['total_doubles'],
        metrics['total_triples'],
        metrics['total_tetrises']
    ]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax5.bar(clear_types, clear_counts, color=colors, edgecolor='black', alpha=0.7)
    ax5.set_ylabel('Total Count')
    ax5.set_title('Line Clear Breakdown')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 6. Action distribution
    ax6 = fig.add_subplot(gs[1, 1])
    action_types = ['Left', 'Right', 'Rotate', 'Soft\nDrop', 'Hard\nDrop', 'No-op']
    action_counts = [
        metrics['total_left_moves'],
        metrics['total_right_moves'],
        metrics['total_rotations'],
        metrics['total_soft_drops'],
        metrics['total_hard_drops'],
        metrics['total_no_ops']
    ]
    ax6.bar(action_types, action_counts, edgecolor='black', alpha=0.7)
    ax6.set_ylabel('Total Count')
    ax6.set_title('Action Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), fontsize=8)
    
    # 7. Score vs Lines scatter
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(df['lines_cleared'], df['score'], alpha=0.3, s=10)
    ax7.set_xlabel('Lines Cleared')
    ax7.set_ylabel('Score')
    ax7.set_title('Score vs Lines Cleared')
    ax7.grid(True, alpha=0.3)
    
    # 8. Max height distribution
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(df['max_height'], bins=20, edgecolor='black', alpha=0.7, color='brown')
    ax8.axvline(x=metrics['avg_max_height'], color='r', linestyle='--', 
                label=f"Mean: {metrics['avg_max_height']:.1f}")
    ax8.axvline(x=15, color='orange', linestyle='--', alpha=0.5, label='Danger (15)')
    ax8.set_xlabel('Max Height')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Max Height Distribution')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Holes created distribution
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(df['holes_created'], bins=20, edgecolor='black', alpha=0.7, color='red')
    ax9.axvline(x=metrics['avg_holes'], color='darkred', linestyle='--', 
                label=f"Mean: {metrics['avg_holes']:.1f}")
    ax9.set_xlabel('Holes Created')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Holes Distribution')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. Invalid action rate
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(df['invalid_action_rate'], bins=20, edgecolor='black', alpha=0.7, color='darkred')
    ax10.axvline(x=metrics['avg_invalid_action_rate'], color='r', linestyle='--', 
                 label=f"Mean: {metrics['avg_invalid_action_rate']:.2%}")
    ax10.set_xlabel('Invalid Action Rate')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Invalid Actions Rate')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Tetris rate distribution
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.hist(df['tetris_rate'], bins=20, edgecolor='black', alpha=0.7, color='darkgreen')
    ax11.axvline(x=metrics['avg_tetris_rate'], color='r', linestyle='--', 
                 label=f"Mean: {metrics['avg_tetris_rate']:.2%}")
    ax11.set_xlabel('Tetris Rate')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Tetris (4-line) Rate Distribution')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. Max combo distribution
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.hist(df['max_combo'], bins=range(int(df['max_combo'].max()) + 2), 
              edgecolor='black', alpha=0.7, color='magenta')
    ax12.set_xlabel('Max Combo')
    ax12.set_ylabel('Frequency')
    ax12.set_title('Max Combo Distribution')
    ax12.grid(True, alpha=0.3, axis='y')
    
    # 13. Performance over episodes
    ax13 = fig.add_subplot(gs[3, :2])
    window = min(50, len(df) // 10)
    df['score_rolling'] = df['score'].rolling(window=window, min_periods=1).mean()
    ax13.plot(df['episode'], df['score_rolling'], linewidth=2, label='Score')
    ax13.set_xlabel('Episode')
    ax13.set_ylabel('Score (Rolling Average)')
    ax13.set_title(f'Score Over Evaluation (Rolling {window} episodes)')
    ax13.grid(True, alpha=0.3)
    ax13.legend()
    
    # 14. Issues summary (text box)
    ax14 = fig.add_subplot(gs[3, 2:])
    ax14.axis('off')
    
    issues = metrics['issues_detected']
    if issues:
        issue_text = "Issues Detected:\n\n"
        for i, issue in enumerate(issues[:5], 1):  # Show top 5
            issue_text += f"{i}. [{issue['severity']}] {issue['type']}\n"
            issue_text += f"   {issue['description'][:60]}...\n\n"
    else:
        issue_text = "✓ No critical issues detected!\n\n"
        issue_text += f"Performance Summary:\n"
        issue_text += f"• Avg Score: {metrics['avg_score']:.1f}\n"
        issue_text += f"• Avg Lines: {metrics['avg_lines']:.1f}\n"
        issue_text += f"• Efficiency: {metrics['avg_lines_per_piece']:.2f} lines/piece\n"
        issue_text += f"• Tetris Rate: {metrics['avg_tetris_rate']:.1%}\n"
    
    ax14.text(0.05, 0.95, issue_text, transform=ax14.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.savefig(save_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation plots saved to {save_path / 'evaluation_results.png'}")
    plt.close()


def compare_models(results: List[Dict], save_path: Path):
    """Generate comparison plots for multiple Tetris models"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Tetris Model Comparison', fontsize=16)
    
    model_names = [r['name'] for r in results]
    
    # 1. Average scores
    ax1 = axes[0, 0]
    scores = [r['metrics']['avg_score'] for r in results]
    bars = ax1.bar(model_names, scores, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average Score Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{score:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Average lines cleared
    ax2 = axes[0, 1]
    lines = [r['metrics']['avg_lines'] for r in results]
    bars = ax2.bar(model_names, lines, alpha=0.7, edgecolor='black', color='green')
    ax2.set_ylabel('Average Lines')
    ax2.set_title('Average Lines Cleared')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    for bar, line in zip(bars, lines):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{line:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Lines per piece efficiency
    ax3 = axes[0, 2]
    efficiency = [r['metrics']['avg_lines_per_piece'] for r in results]
    bars = ax3.bar(model_names, efficiency, alpha=0.7, edgecolor='black', color='purple')
    ax3.set_ylabel('Lines per Piece')
    ax3.set_title('Efficiency Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    for bar, eff in zip(bars, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{eff:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Survival time
    ax4 = axes[1, 0]
    survival = [r['metrics']['avg_survival_time'] for r in results]
    ax4.bar(model_names, survival, alpha=0.7, edgecolor='black', color='orange')
    ax4.set_ylabel('Average Steps')
    ax4.set_title('Survival Time')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 5. Tetris rate
    ax5 = axes[1, 1]
    tetris_rates = [r['metrics']['avg_tetris_rate'] * 100 for r in results]
    ax5.bar(model_names, tetris_rates, alpha=0.7, edgecolor='black', color='darkgreen')
    ax5.set_ylabel('Tetris Rate (%)')
    ax5.set_title('4-Line Clear Rate')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 6. Invalid action rate
    ax6 = axes[1, 2]
    invalid_rates = [r['metrics']['avg_invalid_action_rate'] * 100 for r in results]
    ax6.bar(model_names, invalid_rates, alpha=0.7, edgecolor='black', color='red')
    ax6.set_ylabel('Invalid Action Rate (%)')
    ax6.set_title('Invalid Actions')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 7. Average holes
    ax7 = axes[2, 0]
    holes = [r['metrics']['avg_holes'] for r in results]
    ax7.bar(model_names, holes, alpha=0.7, edgecolor='black', color='brown')
    ax7.set_ylabel('Average Holes')
    ax7.set_title('Holes Created')
    ax7.grid(True, alpha=0.3, axis='y')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 8. Max height
    ax8 = axes[2, 1]
    heights = [r['metrics']['avg_max_height'] for r in results]
    ax8.bar(model_names, heights, alpha=0.7, edgecolor='black', color='navy')
    ax8.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Danger Zone')
    ax8.set_ylabel('Average Max Height')
    ax8.set_title('Height Management')
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.legend(fontsize=8)
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 9. Summary table
    ax9 = axes[2, 2]
    ax9.axis('tight')
    ax9.axis('off')
    
    table_data = []
    for r in results:
        m = r['metrics']
        table_data.append([
            r['name'][:15],  # Truncate long names
            f"{m['avg_score']:.0f}",
            f"{m['avg_lines']:.1f}",
            f"{m['avg_lines_per_piece']:.2f}",
            f"{m['avg_tetris_rate']:.1%}"
        ])
    
    table = ax9.table(
        cellText=table_data,
        colLabels=['Model', 'Score', 'Lines', 'LPP', 'Tetris%'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to {save_path / 'model_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained Tetris DRL agents',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file(s). Use wildcards for multiple models.')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--output-dir', type=str, default='eval_results_tetris',
                       help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models (use wildcards in --model)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes (slows down evaluation)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Import TetrisEnv
    try:
        from tetris_env import TetrisEnv
    except ImportError:
        print("Error: Cannot import TetrisEnv. Make sure tetris_env.py is in PYTHONPATH.")
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
        env = DummyVecEnv([lambda: TetrisEnv(render_mode=None, max_steps=10000)])
        
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
        print(f"Avg Score: {metrics['avg_score']:.1f} (±{metrics['std_score']:.1f})")
        print(f"Max Score: {metrics['max_score']:.0f}")
        print(f"Score P50/P90/P95: {metrics['score_p50']:.0f} / {metrics['score_p90']:.0f} / {metrics['score_p95']:.0f}")
        print(f"\nAvg Lines: {metrics['avg_lines']:.1f} (±{metrics['std_lines']:.1f})")
        print(f"Max Lines: {metrics['max_lines']:.0f}")
        print(f"Lines P50/P90/P95: {metrics['lines_p50']:.0f} / {metrics['lines_p90']:.0f} / {metrics['lines_p95']:.0f}")
        print(f"\nAvg Survival Time: {metrics['avg_survival_time']:.0f} steps")
        print(f"Avg Pieces Placed: {metrics['avg_pieces_placed']:.1f}")
        print(f"Lines per Piece: {metrics['avg_lines_per_piece']:.3f}")
        print(f"Score per Piece: {metrics['avg_score_per_piece']:.2f}")
        print(f"\nTetris Rate: {metrics['avg_tetris_rate']:.2%}")
        print(f"Invalid Action Rate: {metrics['avg_invalid_action_rate']:.2%}")
        print(f"Avg Holes: {metrics['avg_holes']:.1f}")
        print(f"Avg Max Height: {metrics['avg_max_height']:.1f}")
        
        print(f"\nLine Clear Breakdown:")
        print(f"  Singles:  {metrics['total_singles']:,}")
        print(f"  Doubles:  {metrics['total_doubles']:,}")
        print(f"  Triples:  {metrics['total_triples']:,}")
        print(f"  Tetrises: {metrics['total_tetrises']:,}")
        
        if metrics['issues_detected']:
            print(f"\n⚠ Issues Detected ({len(metrics['issues_detected'])}):")
            for issue in metrics['issues_detected'][:3]:  # Show top 3
                print(f"  [{issue['severity']}] {issue['type']}")
                print(f"    → {issue['description']}")
        else:
            print(f"\n✓ No critical issues detected!")
        
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
                'Avg_Score': m['avg_score'],
                'Max_Score': m['max_score'],
                'Avg_Lines': m['avg_lines'],
                'Max_Lines': m['max_lines'],
                'Lines_Per_Piece': m['avg_lines_per_piece'],
                'Tetris_Rate': m['avg_tetris_rate'],
                'Avg_Survival': m['avg_survival_time'],
                'Invalid_Rate': m['avg_invalid_action_rate'],
                'Avg_Holes': m['avg_holes'],
                'Avg_Height': m['avg_max_height'],
                'Singles': m['total_singles'],
                'Doubles': m['total_doubles'],
                'Triples': m['total_triples'],
                'Tetrises': m['total_tetrises'],
                'Issues_Count': len(m['issues_detected']),
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"✓ Comparison data saved to {output_dir / 'model_comparison.csv'}")
        
        # Generate issue report
        with open(output_dir / 'issues_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TETRIS DRL TESTING - ISSUE DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for r in results:
                f.write(f"\nModel: {r['name']}\n")
                f.write("-" * 80 + "\n")
                
                if r['metrics']['issues_detected']:
                    for issue in r['metrics']['issues_detected']:
                        f.write(f"\n[{issue['severity']}] {issue['type']}\n")
                        f.write(f"Description: {issue['description']}\n")
                        f.write(f"Recommendation: {issue['recommendation']}\n")
                else:
                    f.write("\n✓ No issues detected for this model.\n")
                
                f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Issues report saved to {output_dir / 'issues_report.txt'}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()