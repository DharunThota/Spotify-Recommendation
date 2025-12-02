"""Sequential Pattern Mining for Music Recommendations using PrefixSpan (true implementation)."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import pickle
import os
import math

logger = logging.getLogger(__name__)


class SequentialPatternMiner:
    """Mines listening sequences from Spotify history using a true PrefixSpan algorithm."""
    
    def __init__(
        self,
        min_support: float = 0.3,
        max_gap: Optional[int] = None,
        session_gap_minutes: int = 30,
        max_pattern_length: int = 5,
    ):
        """
        Initialize Sequential Pattern Miner.
        
        Args:
            min_support: Minimum frequency for pattern (0-1), patterns appearing less frequently are filtered
            max_gap: Maximum gap between items in sequence (number of songs). If None -> unlimited gap.
            session_gap_minutes: Time gap to consider new listening session (in minutes)
            max_pattern_length: Maximum length of patterns to mine (inclusive)
        """
        self.min_support = min_support
        # if max_gap <= 0, treat as unlimited (None)
        self.max_gap = None if (max_gap is None or max_gap <= 0) else int(max_gap)
        self.session_gap_minutes = session_gap_minutes
        self.max_pattern_length = max_pattern_length

        self.patterns: List[Tuple[List[str], float]] = []  # List of (pattern, support) tuples
        self.pattern_index: Dict[str, List[Tuple[List[str], float]]] = {}  # song_id -> patterns containing it
        self.transitions = defaultdict(Counter)  # song_a -> Counter(next_song -> count)
        self.next_song_predictions = defaultdict(list)  # song_id -> [(next_song, confidence)]
        self.num_sessions = 0  # Track number of sessions
        self.unique_songs = set()  # Track unique songs
        self._sequences: List[List[str]] = []  # Stored sequences for mining internals
    
    def _extract_sequences_from_history(self, listening_history: List[Dict]) -> List[List[str]]:
        """
        Extract listening sequences from Spotify history.
        Groups songs into sessions based on time gaps.
        
        Args:
            listening_history: List of track data from Spotify API with 'id' and 'played_at'
            
        Returns:
            List of sequences where each sequence is a list of song IDs
        """
        # Handle empty input
        if listening_history is None or (isinstance(listening_history, list) and len(listening_history) == 0):
            return []
        
        # Sort by played_at timestamp
        df = pd.DataFrame(listening_history)
        
        if df.empty:
            return []
        
        # Handle both 'id' and 'song_id' column names
        song_id_col = 'song_id' if 'song_id' in df.columns else 'id'
        
        if 'played_at' not in df.columns:
            logger.warning("No 'played_at' timestamp in listening history")
            # If no timestamps, treat all as one sequence
            return [df[song_id_col].tolist()]
        
        df['played_at'] = pd.to_datetime(df['played_at'])
        df = df.sort_values('played_at')
        
        # Calculate time gaps between consecutive songs
        df['time_gap'] = df['played_at'].diff()
        
        # Mark session boundaries (gap > threshold = new session)
        session_threshold = timedelta(minutes=self.session_gap_minutes)
        df['new_session'] = df['time_gap'] > session_threshold
        df['session_id'] = df['new_session'].cumsum()
        
        # Group by session and extract sequences
        sequences = []
        for session_id, group in df.groupby('session_id'):
            sequence = group[song_id_col].tolist()
            # Only keep sequences with at least 2 songs
            if len(sequence) >= 2:
                sequences.append(sequence)
        
        logger.info(f"Extracted {len(sequences)} listening sequences from {len(df)} tracks")
        return sequences
    
    def _mine_patterns_prefixspan(self, sequences: List[List[str]]) -> List[Tuple[List[str], float]]:
        """
        True PrefixSpan implementation (pattern-growth with projected databases).
        
        Args:
            sequences: List of song ID sequences
            
        Returns:
            List of (pattern, support) tuples
        """
        self._sequences = sequences
        total_sequences = len(sequences)
        if total_sequences == 0:
            return []
        
        # Minimum number of sequences a pattern must appear in
        min_count = math.ceil(self.min_support * total_sequences)
        if min_count <= 0:
            min_count = 1
        
        # We'll store patterns as tuple -> count (count is number of distinct sequences containing pattern)
        found_patterns: Dict[Tuple[str, ...], int] = {}
        
        # Helper: initial projected DB contains all sequences with last matched index = -1
        initial_projected = [(seq_id, -1) for seq_id in range(total_sequences)]
        
        def get_candidates(projected):
            """
            From projected DB (list of (seq_id, last_pos)), return counts and left-most positions
            for candidate items that appear after last_pos within allowed gap.
            
            Returns:
                candidates: dict item -> {'count': int, 'positions': [(seq_id, pos), ...]}
            """
            candidates = {}
            seen_in_sequence = {}  # item -> set(seq_id) to ensure we count per-sequence once
            for seq_id, last_pos in projected:
                seq = sequences[seq_id]
                start_idx = last_pos + 1
                if self.max_gap is None:
                    end_idx = len(seq)
                else:
                    # allowed next_idx <= last_pos + max_gap + 1
                    end_idx = min(len(seq), last_pos + self.max_gap + 2)
                # find all distinct items in this allowed window; but for projection we pick the leftmost occurrence
                local_seen = set()
                for idx in range(start_idx, end_idx):
                    item = seq[idx]
                    if item in local_seen:
                        continue
                    local_seen.add(item)
                    if item not in seen_in_sequence:
                        seen_in_sequence[item] = set()
                    if seq_id not in seen_in_sequence[item]:
                        seen_in_sequence[item].add(seq_id)
                        # record leftmost occurrence for projection: only keep earliest occurrence index per sequence
                        if item not in candidates:
                            candidates[item] = {'count': 0, 'positions': []}
                        candidates[item]['count'] += 1
                        candidates[item]['positions'].append((seq_id, idx))
            return candidates
        
        def recurse(prefix: Tuple[str, ...], projected):
            """
            Recursive PrefixSpan.
            prefix: current pattern tuple
            projected: list of (seq_id, last_pos) indicating sequences and last matched position
            """
            candidates = get_candidates(projected)
            # iterate candidates in deterministic order (e.g., sorted by count desc then item)
            for item, v in sorted(candidates.items(), key=lambda kv: (-kv[1]['count'], kv[0])):
                count = v['count']
                if count < min_count:
                    continue
                new_pattern = prefix + (item,)
                found_patterns[new_pattern] = count
                # Build new projected DB for recursion using recorded leftmost positions
                new_projected = v['positions']
                # Limit pattern length
                if len(new_pattern) < self.max_pattern_length:
                    recurse(new_pattern, new_projected)
        
        # Start recursion with empty prefix
        recurse(tuple(), initial_projected)
        
        # Convert found_patterns to (pattern_list, support)
        patterns = []
        for pattern, count in found_patterns.items():
            support = count / total_sequences
            patterns.append((list(pattern), support))
        
        # Optionally sort by support then by length desc
        patterns.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        logger.info(f"PrefixSpan mined {len(patterns)} patterns (min_count={min_count}, total_sequences={total_sequences})")
        return patterns
    
    def _build_transition_graph(self, sequences: List[List[str]]) -> None:
        """
        Build song-to-song transition graph from sequences.
        
        Args:
            sequences: List of song ID sequences
        """
        self.transitions.clear()
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                current_song = seq[i]
                next_song = seq[i + 1]
                self.transitions[current_song][next_song] += 1
    
    def _build_prediction_index(self) -> None:
        """Build index for fast next-song predictions based on patterns."""
        self.next_song_predictions.clear()
        self.pattern_index.clear()
        
        # Index patterns by songs they contain
        for pattern, support in self.patterns:
            for song_id in pattern:
                if song_id not in self.pattern_index:
                    self.pattern_index[song_id] = []
                self.pattern_index[song_id].append((pattern, support))
        
        # Build next-song predictions from transitions
        for current_song, next_songs in self.transitions.items():
            total = sum(next_songs.values())
            predictions = []
            for next_song, count in next_songs.most_common(20):
                confidence = count / total
                if confidence >= 0.1:  # Only keep predictions with >10% confidence
                    predictions.append((next_song, confidence))
            
            if predictions:
                self.next_song_predictions[current_song] = predictions
    
    def fit(self, listening_history: List[Dict]) -> None:
        """
        Extract sequences from listening history and mine patterns.
        
        Args:
            listening_history: List of recently_played items from Spotify API
                Each item should have: 'id' (song ID) and 'played_at' (timestamp)
        """
        logger.info(f"Mining sequential patterns from {len(listening_history)} listening records")
        
        # Extract sequences from raw history
        sequences = self._extract_sequences_from_history(listening_history)
        
        if not sequences:
            logger.warning("No valid sequences extracted from listening history")
            return
        
        # Track statistics
        self.num_sessions = len(sequences)
        for seq in sequences:
            self.unique_songs.update(seq)
        
        # Mine frequent patterns using true PrefixSpan
        self.patterns = self._mine_patterns_prefixspan(sequences)
        
        # Build transition graph
        self._build_transition_graph(sequences)
        
        # Build prediction indexes
        self._build_prediction_index()
        
        logger.info(f"Pattern mining complete: {len(self.patterns)} patterns, "
                   f"{len(self.next_song_predictions)} songs with predictions")
    
    def predict_next_songs(self, current_sequence: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Given current listening sequence, predict next songs.
        
        Args:
            current_sequence: List of recent song IDs (e.g., last 2-3 songs)
            top_k: Number of predictions to return
            
        Returns:
            List of (song_id, confidence_score) tuples
        """
        if not current_sequence:
            return []
        
        predictions = defaultdict(float)
        
        # Strategy 1: Look for patterns that match current sequence (exact prefix matching inside pattern)
        for pattern, support in self.patterns:
            pattern_len = len(pattern)
            seq_len = len(current_sequence)
            
            # try to find the longest suffix of current_sequence that matches a prefix of pattern
            max_match = 0
            for l in range(1, min(pattern_len - 1, seq_len) + 1):
                if current_sequence[-l:] == pattern[:l]:
                    max_match = l
            if max_match > 0 and max_match < pattern_len:
                next_song = pattern[max_match]
                # Score combines pattern support and match quality
                score = support * (max_match / (pattern_len - 1))
                predictions[next_song] = max(predictions[next_song], score)
        
        # Strategy 2: Use direct transitions from last song
        last_song = current_sequence[-1]
        if last_song in self.next_song_predictions:
            for next_song, confidence in self.next_song_predictions[last_song]:
                # Combine with existing prediction or add new
                predictions[next_song] = max(predictions[next_song], confidence * 0.8)
        
        # Sort by score and return top-k
        ranked_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return ranked_predictions[:top_k]
    
    def get_pattern_explanation(self, song_id: str, context_songs: List[str]) -> Dict:
        """
        Explain why a song was recommended based on sequence patterns.
        
        Args:
            song_id: The recommended song ID
            context_songs: The context sequence that led to this recommendation
            
        Returns:
            Dict with explanation details
        """
        explanation = {
            "song_id": song_id,
            "pattern": None,
            "support": 0.0,
            "confidence": 0.0,
            "explanation_text": "",
            "pattern_type": "none"
        }
        
        # Find matching patterns
        matching_patterns = []
        if song_id in self.pattern_index:
            for pattern, support in self.pattern_index[song_id]:
                # Check if context songs appear in pattern before song_id
                try:
                    song_idx = pattern.index(song_id)
                    # Check if any context songs appear before this position
                    for ctx_song in context_songs[::-1]:  # prefer most recent context match
                        if ctx_song in pattern[:song_idx]:
                            matching_patterns.append((pattern, support, ctx_song))
                            break
                except ValueError:
                    continue
        
        if matching_patterns:
            # Use best matching pattern (highest support)
            best_pattern, support, ctx_song = max(matching_patterns, key=lambda x: x[1])
            explanation["pattern"] = best_pattern
            explanation["support"] = support
            explanation["pattern_type"] = "sequential"
            
            # Calculate confidence from transitions (if possible)
            if context_songs and context_songs[-1] in self.transitions:
                trans = self.transitions[context_songs[-1]]
                if song_id in trans:
                    total = sum(trans.values())
                    explanation["confidence"] = trans[song_id] / total
            
            # Generate human-readable explanation
            support_pct = int(support * 100)
            if len(context_songs) > 1:
                explanation["explanation_text"] = (
                    f"{support_pct}% of sessions that contained this sequence also included this track"
                )
            else:
                explanation["explanation_text"] = (
                    f"{support_pct}% of listeners who played {ctx_song} before chose this song later in the session"
                )
        
        # Fallback to direct transition
        elif context_songs and context_songs[-1] in self.next_song_predictions:
            for next_song, confidence in self.next_song_predictions[context_songs[-1]]:
                if next_song == song_id:
                    explanation["confidence"] = confidence
                    explanation["pattern_type"] = "transition"
                    confidence_pct = int(confidence * 100)
                    explanation["explanation_text"] = (
                        f"{confidence_pct}% of listeners who played your last track "
                        f"chose this song next"
                    )
                    break
        
        if not explanation["explanation_text"]:
            explanation["explanation_text"] = "Recommended based on listening patterns"
        
        return explanation
    
    def save(self, filepath: str) -> None:
        """Save mined patterns to file."""
        data = {
            'patterns': self.patterns,
            'transitions': dict(self.transitions),
            'next_song_predictions': dict(self.next_song_predictions),
            'pattern_index': self.pattern_index,
            'num_sessions': self.num_sessions,
            'unique_songs': list(self.unique_songs),
            'config': {
                'min_support': self.min_support,
                'max_gap': self.max_gap,
                'session_gap_minutes': self.session_gap_minutes,
                'max_pattern_length': self.max_pattern_length
            }
        }
        
        # Create directory if filepath contains a directory path
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved sequence patterns to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SequentialPatternMiner':
        """Load mined patterns from file and return a new miner instance."""
        if not os.path.exists(filepath):
            logger.warning(f"Pattern file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance with loaded config
        config = data['config']
        miner = cls(
            min_support=config.get('min_support', 0.3),
            max_gap=config.get('max_gap', None),
            session_gap_minutes=config.get('session_gap_minutes', 30),
            max_pattern_length=config.get('max_pattern_length', 5),
        )
        
        # Load mined data
        miner.patterns = data['patterns']
        miner.transitions = defaultdict(Counter, {k: Counter(v) for k, v in data['transitions'].items()})
        miner.next_song_predictions = defaultdict(list, data['next_song_predictions'])
        miner.pattern_index = data['pattern_index']
        miner.num_sessions = data.get('num_sessions', 0)
        miner.unique_songs = set(data.get('unique_songs', []))
        
        logger.info(f"Loaded {len(miner.patterns)} patterns from {filepath}")
        return miner
    
    def get_statistics(self) -> Dict:
        """Get statistics about mined patterns."""
        pattern_lengths = [len(p) for p, _ in self.patterns] if self.patterns else []
        
        return {
            "total_sessions": self.num_sessions,
            "unique_songs": len(self.unique_songs),
            "patterns_found": len(self.patterns),
            "songs_with_predictions": len(self.next_song_predictions),
            "total_transitions": sum(len(v) for v in self.transitions.values()),
            "avg_pattern_length": np.mean(pattern_lengths) if pattern_lengths else 0,
            "top_patterns": self.patterns[:10]
        }
