import argparse
import time
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import threading
import queue

# Add env directory to Python path
current_dir = Path(__file__).parent.absolute()
env_dir = current_dir.parent / 'env'
sys.path.insert(0, str(env_dir))

from stable_baselines3 import PPO, A2C, DQN
import numpy as np

try:
    from wordle_env import WordleEnv
except ImportError as e:
    print(f"Error: Cannot import WordleEnv. {e}")
    print(f"Looked in: {env_dir}")
    print("Make sure wordle_env.py exists in the env directory.")
    sys.exit(1)


class WordleGUI:
    """Graphical Wordle Visualizer"""
    
    # Color scheme
    COLORS = {
        'correct': '#6aaa64',    # Green
        'present': '#c9b458',    # Yellow
        'absent': '#787c7e',     # Gray
        'unknown': '#ffffff',    # White
        'border': '#d3d6da',     # Light gray
        'bg': '#121213',         # Dark background
        'text': '#ffffff',       # White text
        'key_bg': '#818384',     # Key background
    }
    
    def __init__(self, model, env, args):
        self.model = model
        self.env = env
        self.args = args
        
        # GUI state
        self.paused = False
        self.current_episode = 0
        self.stats = {
            'total_wins': 0,
            'total_games': 0,
            'guess_distribution': {i: 0 for i in range(1, 7)}
        }
        
        # Communication queue
        self.update_queue = queue.Queue()
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Wordle AI Agent Visualizer")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.geometry("800x900")
        
        self._create_widgets()
        self._setup_keyboard()
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="WORDLE AI AGENT",
            font=("Helvetica", 24, "bold"),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        title_label.pack()
        
        # Episode info
        self.info_label = tk.Label(
            title_frame,
            text="Episode 0/0",
            font=("Helvetica", 12),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        self.info_label.pack(pady=5)
        
        # Game board frame
        board_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        board_frame.pack(pady=20)
        
        # Create 6x5 grid
        self.tiles = []
        for row in range(6):
            row_tiles = []
            row_frame = tk.Frame(board_frame, bg=self.COLORS['bg'])
            row_frame.pack(pady=3)
            
            for col in range(5):
                tile = tk.Label(
                    row_frame,
                    text="",
                    font=("Helvetica", 32, "bold"),
                    width=2,
                    height=1,
                    bg=self.COLORS['unknown'],
                    fg=self.COLORS['text'],
                    relief="solid",
                    borderwidth=2,
                    bd=2
                )
                tile.pack(side=tk.LEFT, padx=3)
                row_tiles.append(tile)
            
            self.tiles.append(row_tiles)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready to start...",
            font=("Helvetica", 14),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        self.status_label.pack()
        
        self.reward_label = tk.Label(
            status_frame,
            text="Reward: 0.00",
            font=("Helvetica", 12),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        self.reward_label.pack(pady=3)
        
        self.possible_words_label = tk.Label(
            status_frame,
            text="",
            font=("Helvetica", 11),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        self.possible_words_label.pack(pady=2)
        
        # Keyboard frame
        keyboard_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        keyboard_frame.pack(pady=20)
        
        kb_label = tk.Label(
            keyboard_frame,
            text="KEYBOARD STATE",
            font=("Helvetica", 12, "bold"),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        kb_label.pack(pady=5)
        
        self.keyboard_frame = tk.Frame(keyboard_frame, bg=self.COLORS['bg'])
        self.keyboard_frame.pack()
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        control_frame.pack(pady=10)
        
        self.pause_button = tk.Button(
            control_frame,
            text="Pause",
            font=("Helvetica", 12),
            command=self.toggle_pause,
            width=10,
            bg='#538d4e',
            fg='white',
            relief="raised"
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        stats_label = tk.Label(
            stats_frame,
            text="STATISTICS",
            font=("Helvetica", 12, "bold"),
            bg=self.COLORS['bg'],
            fg=self.COLORS['text']
        )
        stats_label.pack()
        
        self.stats_text = tk.Text(
            stats_frame,
            height=6,
            width=60,
            font=("Courier", 10),
            bg='#1a1a1b',
            fg=self.COLORS['text'],
            relief="solid",
            borderwidth=1
        )
        self.stats_text.pack(pady=5)
        self.stats_text.config(state=tk.DISABLED)
        
    def _setup_keyboard(self):
        """Create virtual keyboard"""
        keyboard_rows = [
            'QWERTYUIOP',
            'ASDFGHJKL',
            'ZXCVBNM'
        ]
        
        self.key_labels = {}
        
        for row_text in keyboard_rows:
            row_frame = tk.Frame(self.keyboard_frame, bg=self.COLORS['bg'])
            row_frame.pack(pady=2)
            
            for letter in row_text:
                key = tk.Label(
                    row_frame,
                    text=letter,
                    font=("Helvetica", 11, "bold"),
                    width=3,
                    height=1,
                    bg=self.COLORS['key_bg'],
                    fg=self.COLORS['text'],
                    relief="raised",
                    borderwidth=1
                )
                key.pack(side=tk.LEFT, padx=2)
                self.key_labels[letter] = key
    
    def reset_board(self):
        """Reset the game board"""
        for row in self.tiles:
            for tile in row:
                tile.config(text="", bg=self.COLORS['unknown'])
        
        # Reset keyboard
        for key in self.key_labels.values():
            key.config(bg=self.COLORS['key_bg'])
    
    def update_tile(self, row, col, letter, feedback):
        """Update a single tile"""
        tile = self.tiles[row][col]
        tile.config(text=letter)
        
        # Set color based on feedback
        if feedback == WordleEnv.CORRECT:
            color = self.COLORS['correct']
        elif feedback == WordleEnv.PRESENT:
            color = self.COLORS['present']
        elif feedback == WordleEnv.ABSENT:
            color = self.COLORS['absent']
        else:
            color = self.COLORS['unknown']
        
        tile.config(bg=color)
    
    def update_keyboard(self, letter_knowledge):
        """Update keyboard colors"""
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for i, letter in enumerate(alphabet):
            if letter in self.key_labels:
                knowledge = letter_knowledge[i]
                
                if knowledge == WordleEnv.CORRECT:
                    color = self.COLORS['correct']
                elif knowledge == WordleEnv.PRESENT:
                    color = self.COLORS['present']
                elif knowledge == WordleEnv.ABSENT:
                    color = self.COLORS['absent']
                else:
                    color = self.COLORS['key_bg']
                
                self.key_labels[letter].config(bg=color)
    
    def update_stats(self):
        """Update statistics display"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        total = self.stats['total_games']
        wins = self.stats['total_wins']
        win_rate = (wins / total * 100) if total > 0 else 0
        
        stats_str = f"Total Games: {total}\n"
        stats_str += f"Wins: {wins} ({win_rate:.1f}%)\n"
        stats_str += f"\nGuess Distribution:\n"
        
        for i in range(1, 7):
            count = self.stats['guess_distribution'][i]
            bar = '█' * count
            stats_str += f"  {i}: {bar} ({count})\n"
        
        self.stats_text.insert(1.0, stats_str)
        self.stats_text.config(state=tk.DISABLED)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            self.status_label.config(text="⏸ PAUSED")
        else:
            self.pause_button.config(text="Pause")
    

    def show_result(self, won, guess_count, target_word):
        """Show game result"""
        if won:
            msg = f"🎉 SUCCESS!\n\nSolved in {guess_count} guesses!"
            self.status_label.config(text=f"✓ Won in {guess_count} guesses!", fg='#6aaa64')
        else:
            msg = f"❌ FAILED!\n\nThe word was: {target_word}"
            self.status_label.config(text=f"✗ Lost! Word was: {target_word}", fg='#f44336')
    
    def process_updates(self):
        """Process updates from the game thread"""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'reset':
                    self.reset_board()
                    self.current_episode = data['episode']
                    self.info_label.config(text=f"Episode {self.current_episode}/{self.args.episodes}")
                    self.status_label.config(text="Agent thinking...", fg=self.COLORS['text'])
                
                elif update_type == 'guess':
                    row = data['row']
                    guess = data['guess']
                    feedback = data['feedback']
                    
                    for col, (letter, fb) in enumerate(zip(guess, feedback)):
                        self.update_tile(row, col, letter, fb)
                
                elif update_type == 'keyboard':
                    self.update_keyboard(data)
                
                elif update_type == 'status':
                    self.status_label.config(text=data['text'], fg=self.COLORS['text'])
                    if 'reward' in data:
                        self.reward_label.config(text=f"Reward: {data['reward']:.2f}")
                    if 'possible_words' in data:
                        self.possible_words_label.config(
                            text=f"Possible words: {data['possible_words']}"
                        )
                
                elif update_type == 'result':
                    self.show_result(data['won'], data['guess_count'], data['target_word'])
                    self.stats['total_games'] += 1
                    if data['won']:
                        self.stats['total_wins'] += 1
                        self.stats['guess_distribution'][data['guess_count']] += 1
                    self.update_stats()
                
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self.process_updates)
    
    def run(self):
        """Start the GUI"""
        # Start processing updates
        self.process_updates()
        
        # Start game thread
        game_thread = threading.Thread(target=self.run_episodes, daemon=True)
        game_thread.start()
        
        # Run GUI main loop
        self.root.mainloop()
    
    def run_episodes(self):
        """Run episodes in background thread"""
        for episode in range(self.args.episodes):
            # Reset environment
            obs, info = self.env.reset()
            target_word = info['target_word']
            
            # Signal GUI to reset
            self.update_queue.put(('reset', {'episode': episode + 1}))
            time.sleep(0.5)
            
            done = False
            truncated = False
            guess_num = 0
            skip = False
            
            while not (done or truncated):
                # Wait if paused
                while self.paused:
                    time.sleep(0.1)
                
                guess_num += 1
                
                # Show thinking status
                self.update_queue.put(('status', {'text': '🤔 Agent thinking...'}))
                
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = self.env.step(int(action))
                
                # Get guess and feedback
                guess = info.get('last_guess', '')
                feedback_grid = obs['feedback_grid']
                feedback = feedback_grid[guess_num - 1]
                letter_knowledge = obs['letter_knowledge']
                
                # Update GUI
                self.update_queue.put(('guess', {
                    'row': guess_num - 1,
                    'guess': guess,
                    'feedback': feedback
                }))
                
                time.sleep(0.3)
                
                self.update_queue.put(('keyboard', letter_knowledge))
                
                # Update status
                status_data = {
                    'text': f'Guess {guess_num}/6',
                    'reward': reward
                }
                if 'possible_words_count' in info:
                    status_data['possible_words'] = info['possible_words_count']
                
                self.update_queue.put(('status', status_data))
                
                # Delay between guesses
                time.sleep(self.args.delay)
            
            if not skip:
                # Show result
                self.update_queue.put(('result', {
                    'won': info.get('won', False),
                    'guess_count': guess_num,
                    'target_word': target_word
                }))
                
                # Wait before next episode
                if episode < self.args.episodes - 1:
                    time.sleep(2)


def load_model(model_path: str):
    """Load a trained model"""
    model_path = Path(model_path)
    
    if 'ppo' in str(model_path).lower():
        return PPO.load(model_path)
    elif 'a2c' in str(model_path).lower():
        return A2C.load(model_path)
    elif 'dqn' in str(model_path).lower():
        return DQN.load(model_path)
    else:
        for algo_class in [PPO, A2C, DQN]:
            try:
                return algo_class.load(model_path)
            except:
                continue
        raise ValueError(f"Could not load model from {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trained Wordle agent with GUI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between guesses (seconds)')
    parser.add_argument('--persona', type=str, default='explorer',
                       choices=['explorer', 'speedrunner', 'validator', 'survivor'],
                       help='Environment persona (should match training)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    try:
        model = load_model(args.model)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Create environment
    env = WordleEnv(
        persona=args.persona,
        use_action_masking=True,
        max_guesses=6
    )
    
    # Set seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)
    
    # Create and run GUI
    gui = WordleGUI(model, env, args)
    gui.run()
    
    env.close()


if __name__ == "__main__":
    main()