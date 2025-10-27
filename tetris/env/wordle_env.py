import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Set
from collections import deque, defaultdict
import time


class WordleRewardSystem:
    """Advanced reward system for Wordle DRL agents"""
    
    def __init__(self, persona: str = 'explorer'):
        self.persona = persona
        self.guess_history = []
        self.feedback_history = []
        self.letters_found = set()
        self.letters_eliminated = set()
        self.position_info = defaultdict(set)
        self.total_guesses = 0
        self.weights = self._get_persona_weights(persona)
        
    def _get_persona_weights(self, persona: str) -> Dict[str, float]:
        """Load reward weights from config file"""
        import os
        import yaml
        
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'persona_{persona}.yaml')
        if not os.path.exists(config_path):
            persona = 'explorer'
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'persona_{persona}.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['persona_weights']
            config = yaml.safe_load(f)
            return config['persona_weights']
    
    def reset(self):
        """Reset state for a new episode"""
        self.guess_history = []
        self.feedback_history = []
        self.letters_found = set()
        self.letters_eliminated = set()
        self.position_info = defaultdict(set)
        self.total_guesses = 0
    
    def calculate_reward(
        self,
        guess: str,
        feedback: np.ndarray,
        won: bool,
        lost: bool,
        is_valid_word: bool = True,
        remaining_guesses: int = 6
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive reward for a guess"""
        self.total_guesses += 1
        reward_breakdown = {}
        
        # Terminal state rewards
        if won:
            efficiency_bonus = (7 - self.total_guesses) * self.weights['efficiency_multiplier']
            win_reward = self.weights['win_bonus'] + efficiency_bonus
            reward_breakdown['win_bonus'] = self.weights['win_bonus']
            reward_breakdown['efficiency_bonus'] = efficiency_bonus
            return win_reward, reward_breakdown
        
        if lost:
            reward_breakdown['loss_penalty'] = self.weights['loss_penalty']
            return self.weights['loss_penalty'], reward_breakdown
        
        # Invalid word penalty
        if not is_valid_word:
            reward_breakdown['invalid_word'] = self.weights['invalid_guess_penalty']
            return self.weights['invalid_guess_penalty'], reward_breakdown
        
        # Repeated guess penalty
        if guess in self.guess_history:
            reward_breakdown['repeated_guess'] = self.weights['repeated_guess_penalty']
            return self.weights['repeated_guess_penalty'], reward_breakdown
        
        # Base guess penalty
        guess_penalty = self.weights['guess_penalty_base'] * (1 + 0.1 * self.total_guesses)
        reward_breakdown['guess_penalty'] = guess_penalty
        
        # Information gain reward
        info_reward = self._calculate_information_gain(guess, feedback)
        reward_breakdown['information_gain'] = info_reward
        
        # Letter discovery reward
        letter_reward = self._calculate_letter_discovery(guess, feedback)
        reward_breakdown['letter_discovery'] = letter_reward
        
        # Position discovery reward
        position_reward = self._calculate_position_discovery(guess, feedback)
        reward_breakdown['position_discovery'] = position_reward
        
        # Diversity bonus
        diversity_reward = self._calculate_diversity_bonus(guess)
        reward_breakdown['diversity_bonus'] = diversity_reward
        
        # Progress reward (greens)
        progress_reward = np.sum(feedback == 3) * self.weights['progress_reward']  # CORRECT = 3
        reward_breakdown['progress_reward'] = progress_reward
        
        # Strategy reward
        strategy_reward = self._calculate_strategy_reward(remaining_guesses, feedback)
        reward_breakdown['strategy_reward'] = strategy_reward
        
        # Update internal state
        self._update_state(guess, feedback)
        
        total_reward = sum(reward_breakdown.values())
        return total_reward, reward_breakdown
    
    def _calculate_information_gain(self, guess: str, feedback: np.ndarray) -> float:
        """Reward for gaining new information"""
        gray_count = np.sum(feedback == 1)   # ABSENT = 1
        yellow_count = np.sum(feedback == 2) # PRESENT = 2
        green_count = np.sum(feedback == 3)  # CORRECT = 3
        info_score = (green_count * 3.0 + yellow_count * 2.0 + gray_count * 0.5)
        return info_score * self.weights['info_gain_reward'] / 5.0
    
    def _calculate_letter_discovery(self, guess: str, feedback: np.ndarray) -> float:
        """Reward for discovering new letters"""
        new_letters_found = 0
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb > 1 and letter not in self.letters_found:  # PRESENT or CORRECT
                new_letters_found += 1
        return new_letters_found * self.weights['letter_discovery']
    
    def _calculate_position_discovery(self, guess: str, feedback: np.ndarray) -> float:
        """Reward for discovering correct positions"""
        new_positions_found = 0
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == 3 and letter not in self.position_info[i]:  # CORRECT = 3
                new_positions_found += 1
        return new_positions_found * self.weights['position_discovery']
    
    def _calculate_diversity_bonus(self, guess: str) -> float:
        """Reward for using diverse letters early"""
        if self.total_guesses > 3:
            return 0.0
        unique_new_letters = len(set(guess) - self._all_guessed_letters())
        diversity_multiplier = 1.0 / self.total_guesses
        return unique_new_letters * self.weights['diversity_bonus'] * diversity_multiplier
    
    def _calculate_strategy_reward(self, remaining_guesses: int, feedback: np.ndarray) -> float:
        """Reward strategic play based on game state"""
        green_count = np.sum(feedback == 3)  # CORRECT = 3
        
        if remaining_guesses >= 4:
            if np.any(feedback > 1):  # Any PRESENT or CORRECT
                return 1.0
        elif remaining_guesses >= 2:
            return green_count * 1.5
        else:
            return green_count * 3.0
        return 0.0
    
    def _update_state(self, guess: str, feedback: np.ndarray):
        """Update internal state tracking"""
        self.guess_history.append(guess)
        self.feedback_history.append(feedback)
        
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == 3:  # CORRECT
                self.letters_found.add(letter)
                self.position_info[i].add(letter)
            elif fb == 2:  # PRESENT
                self.letters_found.add(letter)
            elif fb == 1:  # ABSENT
                if letter not in self.letters_found:
                    self.letters_eliminated.add(letter)
    
    def _all_guessed_letters(self) -> Set[str]:
        """Get all letters guessed so far"""
        return set(''.join(self.guess_history))
    
    def get_statistics(self) -> Dict:
        """Get current episode statistics"""
        return {
            'total_guesses': self.total_guesses,
            'letters_found': len(self.letters_found),
            'letters_eliminated': len(self.letters_eliminated),
            'positions_known': sum(len(letters) for letters in self.position_info.values()),
            'unique_guesses': len(set(self.guess_history)),
            'repeated_guesses': len(self.guess_history) - len(set(self.guess_history))
        }


class WordleEnv(gym.Env):
    """
    Wordle Environment with Integrated Advanced Reward System
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    # Letter feedback encoding
    UNKNOWN = 0
    ABSENT = 1
    PRESENT = 2
    CORRECT = 3
    
    DEFAULT_WORDS = [
        'AUDIO', 'ADIEU', 'ARISE', 'RAISE', 'IRATE', 'LATER', 'SANER', 'CRATE',
        'SLATE', 'CRANE', 'SLANT', 'TRACE', 'SNARE', 'STARE', 'AROSE', 'TEARS',
        'ABOUT', 'AFTER', 'AGREE', 'ALERT', 'ALLOW', 'ALONE', 'ALONG', 'ANGEL',
        'ANGER', 'ANGLE', 'ANGRY', 'APART', 'APPLE', 'APPLY', 'ARENA', 'ARGUE',
        'ARRAY', 'ASIDE', 'ASSET', 'BAKER', 'BALLS', 'BANDS', 'BANKS', 'BASED',
        'BASIC', 'BASIS', 'BEACH', 'BEGAN', 'BEING', 'BENCH', 'BILLY', 'BIRTH',
        'BLACK', 'BLAME', 'BLIND', 'BLOCK', 'BLOOD', 'BLUES', 'BOARD', 'BOATS',
        'BONDS', 'BONES', 'BOOKS', 'BOOST', 'BOOTH', 'BOUND', 'BRAIN', 'BRAND',
        'BREAD', 'BREAK', 'BREED', 'BRIEF', 'BRING', 'BROAD', 'BROKE', 'BROWN',
        'BUILD', 'BUILT', 'BUYER', 'CABLE', 'CALLS', 'CAMPS', 'CARDS', 'CARGO',
        'CAROL', 'CARRY', 'CASES', 'CATCH', 'CAUSE', 'CHAIN', 'CHAIR', 'CHAOS',
        'CHARM', 'CHART', 'CHASE', 'CHEAP', 'CHECK', 'CHEST', 'CHIEF', 'CHILD',
        'CHINA', 'CHOSE', 'CIVIL', 'CLAIM', 'CLASS', 'CLEAN', 'CLEAR', 'CLIMB',
        'CLOCK', 'CLOSE', 'CLOTH', 'CLOUD', 'COACH', 'COAST', 'COULD', 'COUNT',
        'COURT', 'COVER', 'CRAFT', 'CRASH', 'CRAZY', 'CREAM', 'CRIME', 'CROSS',
        'CROWD', 'CROWN', 'CRUDE', 'CYCLE', 'DAILY', 'DANCE', 'DATED', 'DEALT',
        'DEATH', 'DEBUT', 'DELAY', 'DEPTH', 'DOING', 'DOUBT', 'DOZEN', 'DRAFT',
        'DRAMA', 'DRANK', 'DRAWN', 'DREAM', 'DRESS', 'DRILL', 'DRINK', 'DRIVE',
        'DROVE', 'DYING', 'EAGER', 'EARLY', 'EARTH', 'EIGHT', 'EMPTY', 'ENEMY',
        'ENJOY', 'ENTRY', 'EQUAL', 'ERROR', 'EVENT', 'EVERY', 'EXACT', 'EXTRA',
        'FAITH', 'FALSE', 'FAULT', 'FIBER', 'FIELD', 'FIFTH', 'FIFTY', 'FIGHT',
        'FINAL', 'FIRST', 'FIXED', 'FLASH', 'FLEET', 'FLOOR', 'FLUID', 'FOCUS',
        'FORCE', 'FORTH', 'FORTY', 'FORUM', 'FOUND', 'FRAME', 'FRANK', 'FRAUD',
        'FRESH', 'FRONT', 'FRUIT', 'FULLY', 'FUNNY', 'GIANT', 'GIVEN', 'GLASS',
        'GLOBE', 'GOING', 'GRACE', 'GRADE', 'GRAND', 'GRANT', 'GRASS', 'GRAVE',
        'GREAT', 'GREEN', 'GROSS', 'GROUP', 'GROWN', 'GUARD', 'GUESS', 'GUEST',
        'GUIDE', 'HAPPY', 'HARRY', 'HEART', 'HEAVY', 'HENCE', 'HENRY', 'HORSE',
        'HOTEL', 'HOUSE', 'HUMAN', 'IDEAL', 'IMAGE', 'INDEX', 'INNER', 'INPUT',
        'ISSUE', 'JAPAN', 'JIMMY', 'JOINT', 'JONES', 'JUDGE', 'KNOWN', 'LABEL',
        'LARGE', 'LASER', 'LAUGH', 'LAYER', 'LEARN', 'LEASE', 'LEAST', 'LEAVE',
        'LEGAL', 'LEVEL', 'LEWIS', 'LIGHT', 'LIMIT', 'LINKS', 'LIVES', 'LOCAL',
        'LOGIC', 'LOOSE', 'LOWER', 'LUCKY', 'LUNCH', 'LYING', 'MAGIC', 'MAJOR',
        'MAKER', 'MARCH', 'MARIA', 'MATCH', 'MAYBE', 'MAYOR', 'MEANT', 'MEDIA',
        'METAL', 'MIGHT', 'MINOR', 'MINUS', 'MIXED', 'MODEL', 'MONEY', 'MONTH',
        'MORAL', 'MOTOR', 'MOUNT', 'MOUSE', 'MOUTH', 'MOVED', 'MOVIE', 'MUSIC',
        'NEEDS', 'NEVER', 'NEWLY', 'NIGHT', 'NOISE', 'NORTH', 'NOTED', 'NOVEL',
        'NURSE', 'OCCUR', 'OCEAN', 'OFFER', 'OFTEN', 'ORDER', 'OTHER', 'OUGHT',
        'PAINT', 'PANEL', 'PAPER', 'PARKS', 'PARTY', 'PEACE', 'PETER', 'PHASE',
        'PHONE', 'PHOTO', 'PIECE', 'PILOT', 'PITCH', 'PLACE', 'PLAIN', 'PLANE',
        'PLANT', 'PLATE', 'POINT', 'POUND', 'POWER', 'PRESS', 'PRICE', 'PRIDE',
        'PRIME', 'PRINT', 'PRIOR', 'PRIZE', 'PROOF', 'PROUD', 'PROVE', 'QUEEN',
        'QUICK', 'QUIET', 'QUITE', 'RADIO', 'RANGE', 'RAPID', 'RATIO', 'REACH',
        'READY', 'REFER', 'RIGHT', 'RIVER', 'ROBIN', 'ROCKY', 'ROGER', 'ROMAN',
        'ROUGH', 'ROUND', 'ROUTE', 'ROYAL', 'RURAL', 'SCALE', 'SCENE', 'SCOPE',
        'SCORE', 'SENSE', 'SERVE', 'SEVEN', 'SHALL', 'SHAPE', 'SHARE', 'SHARP',
        'SHEET', 'SHELF', 'SHELL', 'SHIFT', 'SHINE', 'SHIRT', 'SHOCK', 'SHOOT',
        'SHORT', 'SHOWN', 'SIGHT', 'SINCE', 'SIXTH', 'SIXTY', 'SIZED', 'SKILL',
        'SLEEP', 'SLIDE', 'SMALL', 'SMART', 'SMILE', 'SMITH', 'SMOKE', 'SOLID',
        'SOLVE', 'SORRY', 'SOUND', 'SOUTH', 'SPACE', 'SPARE', 'SPEAK', 'SPEED',
        'SPEND', 'SPENT', 'SPLIT', 'SPOKE', 'SPORT', 'STAFF', 'STAGE', 'STAKE',
        'STAND', 'START', 'STATE', 'STEAM', 'STEEL', 'STICK', 'STILL', 'STOCK',
        'STONE', 'STOOD', 'STORE', 'STORM', 'STORY', 'STRIP', 'STUCK', 'STUDY',
        'STUFF', 'STYLE', 'SUGAR', 'SUITE', 'SUPER', 'SWEET', 'TABLE', 'TAKEN',
        'TASTE', 'TAXES', 'TEACH', 'TENDS', 'TERMS', 'TEXAS', 'THANK', 'THEFT',
        'THEIR', 'THEME', 'THERE', 'THESE', 'THICK', 'THING', 'THINK', 'THIRD',
        'THOSE', 'THREE', 'THREW', 'THROW', 'TIGHT', 'TIMES', 'TITLE', 'TODAY',
        'TOPIC', 'TOTAL', 'TOUCH', 'TOUGH', 'TOWER', 'TRACK', 'TRADE', 'TRAIN',
        'TREAT', 'TREND', 'TRIAL', 'TRIBE', 'TRICK', 'TRIED', 'TRIES', 'TROOP',
        'TRUCK', 'TRULY', 'TRUNK', 'TRUST', 'TRUTH', 'TWICE', 'UNDER', 'UNDUE',
        'UNION', 'UNITY', 'UNTIL', 'UPPER', 'UPSET', 'URBAN', 'USAGE', 'USUAL',
        'VALID', 'VALUE', 'VIDEO', 'VIRUS', 'VISIT', 'VITAL', 'VOCAL', 'VOICE',
        'WASTE', 'WATCH', 'WATER', 'WHEEL', 'WHERE', 'WHICH', 'WHILE', 'WHITE',
        'WHOLE', 'WHOSE', 'WOMAN', 'WOMEN', 'WORLD', 'WORRY', 'WORSE', 'WORST',
        'WORTH', 'WOULD', 'WOUND', 'WRITE', 'WRONG', 'WROTE', 'YIELD', 'YOUNG',
        'YOUTH'
    ]
    
    def __init__(
        self,
        word_size: int = 5,
        persona: str = "explorer",
        max_guesses: int = 6,
        use_action_masking: bool = True,
        render_mode: Optional[str] = None,
        words_list: Optional[list] = None
    ):
        super().__init__()
        
        self.word_size = word_size
        self.persona = persona
        self.max_guesses = max_guesses
        self.use_action_masking = use_action_masking
        self.render_mode = render_mode
        
        # Word list
        self.words_list = [w.upper() for w in (words_list or self.DEFAULT_WORDS)]
        
        if not self.words_list:
            raise ValueError("words_list cannot be empty")
        
        # Action space
        self.action_space = spaces.Discrete(len(self.words_list))
        
        # Observation space
        self.observation_space = spaces.Dict({
            'feedback_grid': spaces.Box(
                low=0, high=3,
                shape=(self.max_guesses, self.word_size),
                dtype=np.int8
            ),
            'letter_knowledge': spaces.Box(
                low=0, high=3,
                shape=(26,),
                dtype=np.int8
            ),
            'guess_count': spaces.Box(
                low=0, high=self.max_guesses,
                shape=(1,),
                dtype=np.int8
            )
        })
        
        # Initialize reward system
        self.reward_system = WordleRewardSystem(persona=persona)
        
        # Episode state
        self.target_word = ""
        self.guess_count = 0
        self.won = False
        self.guesses_made = []
        self.feedback_grid = np.zeros((self.max_guesses, self.word_size), dtype=np.int8)
        self.letter_knowledge = np.zeros(26, dtype=np.int8)
        self.possible_words = set()
        
        # Metrics
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.win_rate_window = deque(maxlen=100)
        self.total_episodes = 0
        self.total_wins = 0
        self.total_guesses = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Select new target
        self.target_word = np.random.choice(self.words_list)
        
        # Reset state
        self.guess_count = 0
        self.won = False
        self.guesses_made = []
        self.feedback_grid = np.zeros((self.max_guesses, self.word_size), dtype=np.int8)
        self.letter_knowledge = np.zeros(26, dtype=np.int8)
        self.possible_words = set(self.words_list)
        
        self.episode_start_time = time.time()
        self.episode_reward = 0.0
        
        # Reset reward system
        self.reward_system.reset()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _letter_to_idx(self, letter: str) -> int:
        """Convert letter to index"""
        return ord(letter.upper()) - ord('A')
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        return {
            'feedback_grid': self.feedback_grid.copy(),
            'letter_knowledge': self.letter_knowledge.copy(),
            'guess_count': np.array([self.guess_count], dtype=np.int8)
        }
    
    def _calculate_feedback(self, guess: str) -> np.ndarray:
        """Calculate Wordle feedback"""
        guess = guess.upper()
        target = self.target_word
        feedback = np.zeros(self.word_size, dtype=np.int8)
        
        target_letters = list(target)
        guess_letters = list(guess)
        
        # First pass: mark correct (green)
        for i in range(self.word_size):
            if guess_letters[i] == target_letters[i]:
                feedback[i] = self.CORRECT
                target_letters[i] = None
                guess_letters[i] = None
        
        # Second pass: mark present (yellow)
        for i in range(self.word_size):
            if guess_letters[i] is not None:
                if guess_letters[i] in target_letters:
                    feedback[i] = self.PRESENT
                    target_letters[target_letters.index(guess_letters[i])] = None
                else:
                    feedback[i] = self.ABSENT
        
        return feedback
    
    def _update_letter_knowledge(self, guess: str, feedback: np.ndarray):
        """Update letter knowledge"""
        for i, letter in enumerate(guess.upper()):
            letter_idx = self._letter_to_idx(letter)
            feedback_val = feedback[i]
            if feedback_val > self.letter_knowledge[letter_idx]:
                self.letter_knowledge[letter_idx] = feedback_val
    
    def _update_possible_words(self, guess: str, feedback: np.ndarray):
        """Filter possible words"""
        if not self.use_action_masking:
            return
        
        guess = guess.upper()
        new_possible = set()
        
        for word in self.possible_words:
            word = word.upper()
            if self._is_consistent_with_feedback(word, guess, feedback):
                new_possible.add(word)
        
        self.possible_words = new_possible
    
    def _is_consistent_with_feedback(
        self,
        candidate: str,
        guess: str,
        feedback: np.ndarray
    ) -> bool:
        """Check consistency with feedback"""
        # Check CORRECT positions
        for i in range(self.word_size):
            if feedback[i] == self.CORRECT:
                if candidate[i] != guess[i]:
                    return False
        
        # Build letter count requirements
        letter_min_count = {}
        letter_max_count = {}
        
        for i in range(self.word_size):
            letter = guess[i]
            
            if feedback[i] == self.CORRECT or feedback[i] == self.PRESENT:
                letter_min_count[letter] = letter_min_count.get(letter, 0) + 1
            
            elif feedback[i] == self.ABSENT:
                has_match_elsewhere = any(
                    (feedback[j] == self.CORRECT or feedback[j] == self.PRESENT) 
                    and guess[j] == letter
                    for j in range(self.word_size)
                )
                
                if not has_match_elsewhere:
                    letter_max_count[letter] = 0
                else:
                    if letter not in letter_max_count:
                        count = sum(
                            1 for j in range(self.word_size)
                            if guess[j] == letter and 
                            (feedback[j] == self.CORRECT or feedback[j] == self.PRESENT)
                        )
                        letter_max_count[letter] = count
        
        # Check candidate
        for letter, min_count in letter_min_count.items():
            if candidate.count(letter) < min_count:
                return False
        
        for letter, max_count in letter_max_count.items():
            if candidate.count(letter) > max_count:
                return False
        
        # Check PRESENT constraints
        for i in range(self.word_size):
            if feedback[i] == self.PRESENT:
                if candidate[i] == guess[i]:
                    return False
        
        return True
    
    def _get_action_mask(self) -> np.ndarray:
        """Get action mask"""
        if not self.use_action_masking:
            return np.ones(self.action_space.n, dtype=np.int8)
        
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        for idx, word in enumerate(self.words_list):
            if word.upper() in self.possible_words:
                mask[idx] = 1
        
        if mask.sum() == 0:
            mask[:] = 1
        
        return mask
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        info = {
            'guess_count': self.guess_count,
            'won': self.won,
            'target_word': self.target_word,
        }
        
        if self.guesses_made:
            info['last_guess'] = self.guesses_made[-1]
        
        if self.use_action_masking:
            info['action_mask'] = self._get_action_mask()
            info['possible_words_count'] = len(self.possible_words)
        
        return info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step"""
        # Validate action
        if action < 0 or action >= self.action_space.n:
            return (
                self._get_observation(),
                -1.0,
                True,
                False,
                {'error': 'invalid_action', 'action': action}
            )
        
        # Get guess
        guess = self.words_list[action].upper()
        
        # Calculate feedback
        feedback = self._calculate_feedback(guess)
        
        # Update state
        self.guess_count += 1
        self.guesses_made.append(guess)
        self.feedback_grid[self.guess_count - 1] = feedback
        self._update_letter_knowledge(guess, feedback)
        self._update_possible_words(guess, feedback)
        
        # Check win/loss
        if guess == self.target_word:
            self.won = True
            done = True
        elif self.guess_count >= self.max_guesses:
            done = True
        else:
            done = False
        
        # Calculate reward using reward system
        remaining_guesses = self.max_guesses - self.guess_count
        reward, breakdown = self.reward_system.calculate_reward(
            guess=guess,
            feedback=feedback,
            won=self.won,
            lost=done and not self.won,
            is_valid_word=True,  # Already validated by action space
            remaining_guesses=remaining_guesses
        )
        
        self.episode_reward += reward
        
        # Update metrics on episode end
        if done:
            self.total_episodes += 1
            if self.won:
                self.total_wins += 1
                self.win_rate_window.append(1)
            else:
                self.win_rate_window.append(0)
            
            self.total_guesses += self.guess_count
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.guess_count)
        
        obs = self._get_observation()
        info = self._get_info()
        
        # Add reward breakdown to info
        info['reward_breakdown'] = breakdown
        info['reward_stats'] = self.reward_system.get_statistics()
        
        # Add episode summary if done
        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.guess_count,
                'won': self.won
            }
        
        return obs, reward, done, False, info
    
    def render(self):
        """Render the game state"""
        if self.render_mode is None:
            return
        
        output = []
        output.append(f"\n{'='*40}")
        output.append(f"Wordle Game - Guess {self.guess_count}/{self.max_guesses}")
        output.append(f"Persona: {self.persona}")
        output.append(f"Target: {'*' * self.word_size} (hidden)")
        output.append(f"{'='*40}\n")
        
        # Feedback symbols
        symbols = {
            self.UNKNOWN: '·',
            self.ABSENT: '⬛',
            self.PRESENT: '🟨',
            self.CORRECT: '🟩'
        }
        
        for i in range(self.guess_count):
            guess = self.guesses_made[i]
            feedback = self.feedback_grid[i]
            
            output.append(f"  {guess}  ")
            feedback_str = ''.join(symbols[f] for f in feedback)
            output.append(f"  {feedback_str}\n")
        
        # Remaining guesses
        for i in range(self.guess_count, self.max_guesses):
            output.append(f"  {'-' * self.word_size}\n")
        
        if self.won:
            output.append(f"\n🎉 Won in {self.guess_count} guesses!")
        elif self.guess_count >= self.max_guesses:
            output.append(f"\n❌ Game over! Word was: {self.target_word}")
        
        output.append(f"\n{'='*40}\n")
        
        result = '\n'.join(output)
        
        if self.render_mode == 'human':
            print(result)
        
        return result
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics"""
        metrics = {
            'total_episodes': self.total_episodes,
            'total_wins': self.total_wins,
            'total_guesses': self.total_guesses,
            'win_rate': self.total_wins / max(1, self.total_episodes),
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }
        
        if self.win_rate_window:
            metrics['recent_win_rate'] = np.mean(self.win_rate_window)
        
        if self.total_wins > 0:
            metrics['avg_guesses_when_won'] = self.total_guesses / self.total_wins
        
        return metrics
    
    def close(self):
        """Cleanup"""
        pass