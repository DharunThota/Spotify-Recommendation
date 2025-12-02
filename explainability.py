"""Explainability module for generating human-readable recommendation explanations."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import config
import logging
from sequence_mining import SequentialPatternMiner

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Generates explanations for recommendations."""
    
    def __init__(self, data_processor, recommendation_engine):
        self.processor = data_processor
        self.engine = recommendation_engine
        
    def explain_song_recommendation(
        self, 
        input_song_id: str, 
        recommended_idx: int
    ) -> Dict[str, any]:
        """
        Generate explanation for why a song was recommended.
        
        Args:
            input_song_id: Input song ID
            recommended_idx: Index of recommended song
            
        Returns:
            Dictionary with explanation components
        """
        input_idx = self.processor.song_id_to_idx.get(input_song_id)
        if input_idx is None:
            return {}
        
        input_song = self.processor.get_song_by_index(input_idx)
        recommended_song = self.processor.get_song_by_index(recommended_idx)
        
        # Calculate similarity score
        similarity_score = self.engine.compute_similarity(input_idx, recommended_idx)
        
        # Analyze feature similarities
        feature_analysis = self._analyze_feature_similarities(input_song, recommended_song)
        
        # Check for common attributes
        common_attributes = self._find_common_attributes(input_song, recommended_song)
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            input_song, 
            recommended_song, 
            similarity_score,
            feature_analysis,
            common_attributes
        )
        
        return {
            'similarity_score': float(similarity_score),
            'feature_analysis': feature_analysis,
            'common_attributes': common_attributes,
            'explanation': explanation_text,
            'input_song': {
                'name': input_song['name'],
                'artists': input_song['artists'],
                'year': int(input_song['year'])
            },
            'recommended_song': {
                'name': recommended_song['name'],
                'artists': recommended_song['artists'],
                'year': int(recommended_song['year']),
                'popularity': int(recommended_song['popularity'])
            }
        }
    
    def explain_mood_recommendation(
        self, 
        mood: str, 
        recommended_idx: int
    ) -> Dict[str, any]:
        """
        Generate explanation for mood-based recommendation.
        
        Args:
            mood: Target mood
            recommended_idx: Index of recommended song
            
        Returns:
            Dictionary with explanation components
        """
        recommended_song = self.processor.get_song_by_index(recommended_idx)
        logger.debug(f"Explaining mood recommendation: idx={recommended_idx}, mood={mood}")
        
        # Calculate mood fit score
        mood_fit = self.engine._calculate_mood_fit(recommended_song, mood)
        logger.debug(f"Mood fit calculated: {mood_fit} (type: {type(mood_fit)})")
        
        # Analyze which features contribute to mood
        mood_features = self._analyze_mood_features(recommended_song, mood)
        logger.debug(f"Mood features analyzed: {len(mood_features)} features")
        
        # Generate explanation
        explanation_text = self._generate_mood_explanation(
            recommended_song,
            mood,
            mood_fit,
            mood_features
        )
        
        return {
            'mood': mood,
            'mood_fit_score': float(mood_fit),
            'mood_features': mood_features,
            'explanation': explanation_text,
            'recommended_song': {
                'name': recommended_song['name'],
                'artists': recommended_song['artists'],
                'year': int(recommended_song['year']),
                'popularity': int(recommended_song['popularity'])
            }
        }
    
    def explain_hybrid_recommendation(
        self,
        input_song_ids: List[str],
        recommended_idx: int,
        mood: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate explanation for hybrid recommendation.
        
        Args:
            input_song_ids: List of input song IDs
            recommended_idx: Index of recommended song
            mood: Optional mood filter
            
        Returns:
            Dictionary with explanation components
        """
        recommended_song = self.processor.get_song_by_index(recommended_idx)
        
        # Get input songs
        input_songs = []
        similarities = []
        
        for song_id in input_song_ids:
            input_idx = self.processor.song_id_to_idx.get(song_id)
            if input_idx is not None:
                input_song = self.processor.get_song_by_index(input_idx)
                similarity = self.engine.compute_similarity(input_idx, recommended_idx)
                input_songs.append(input_song)
                similarities.append(similarity)
        
        if not input_songs:
            return {}
        
        # Calculate average similarity
        avg_similarity = np.mean(similarities)
        
        # Find most similar input song
        most_similar_idx = np.argmax(similarities)
        most_similar_song = input_songs[most_similar_idx]
        
        # Generate explanation
        explanation_text = self._generate_hybrid_explanation(
            input_songs,
            recommended_song,
            most_similar_song,
            avg_similarity,
            mood
        )
        
        return {
            'average_similarity': float(avg_similarity),
            'most_similar_to': {
                'name': most_similar_song['name'],
                'artists': most_similar_song['artists']
            },
            'explanation': explanation_text,
            'mood_filter': mood,
            'recommended_song': {
                'name': recommended_song['name'],
                'artists': recommended_song['artists'],
                'year': int(recommended_song['year']),
                'popularity': int(recommended_song['popularity'])
            }
        }
    
    def _analyze_feature_similarities(self, song1: dict, song2: dict) -> Dict[str, Dict]:
        """Analyze which features are similar between two songs."""
        analysis = {}
        
        feature_labels = {
            'valence': 'positivity',
            'energy': 'energy level',
            'danceability': 'danceability',
            'acousticness': 'acoustic sound',
            'speechiness': 'vocal content',
            'instrumentalness': 'instrumental focus',
            'liveness': 'live performance feel',
            'tempo': 'tempo'
        }
        
        for feature in config.AUDIO_FEATURES:
            if feature in song1 and feature in song2:
                val1 = song1[feature]
                val2 = song2[feature]
                
                # Calculate similarity (normalized difference)
                if feature == 'tempo':
                    # Tempo has different scale
                    diff = abs(val1 - val2) / 200.0  # Normalize by typical tempo range
                elif feature == 'duration_ms':
                    diff = abs(val1 - val2) / 300000.0  # Normalize by 5 minutes
                elif feature == 'loudness':
                    diff = abs(val1 - val2) / 60.0  # Normalize by typical loudness range
                else:
                    diff = abs(val1 - val2)  # Already 0-1 scale
                
                similarity = 1.0 - min(diff, 1.0)
                
                analysis[feature] = {
                    'similarity': float(similarity),
                    'value1': float(val1),
                    'value2': float(val2),
                    'label': feature_labels.get(feature, feature)
                }
        
        # Sort by similarity
        sorted_features = sorted(
            analysis.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )
        
        return dict(sorted_features)
    
    def _find_common_attributes(self, song1: dict, song2: dict) -> Dict[str, any]:
        """Find common attributes between songs."""
        common = {}
        
        # Check for same artists
        artists1 = set(song1['artists_parsed'])
        artists2 = set(song2['artists_parsed'])
        common_artists = artists1.intersection(artists2)
        
        if common_artists:
            common['artists'] = list(common_artists)
        
        # Check for same cluster
        if song1['cluster'] == song2['cluster']:
            common['cluster'] = int(song1['cluster'])
        
        # Check for similar year
        year_diff = abs(song1['year'] - song2['year'])
        if year_diff <= 5:
            common['era'] = f"{song1['year']}s era"
        
        # Check for similar mode and key
        if song1['mode'] == song2['mode']:
            common['mode'] = 'major' if song1['mode'] == 1 else 'minor'
        
        return common
    
    def _analyze_mood_features(self, song: dict, mood: str) -> Dict[str, Dict]:
        """Analyze which features contribute to mood classification."""
        criteria = config.MOOD_CRITERIA[mood]
        analysis = {}
        
        feature_labels = {
            'valence': 'positivity',
            'energy': 'energy',
            'danceability': 'danceability',
            'acousticness': 'acoustic quality',
            'tempo': 'tempo'
        }
        
        for feature, (min_val, max_val) in criteria.items():
            if feature in song:
                value = song[feature]
                
                # Check if in range
                in_range = min_val <= value <= max_val
                
                # Calculate how central the value is
                if in_range:
                    center = (min_val + max_val) / 2
                    range_size = max_val - min_val
                    centrality = 1.0 - (abs(value - center) / (range_size / 2))
                else:
                    centrality = 0.0
                
                analysis[feature] = {
                    'value': float(value),
                    'in_range': in_range,
                    'centrality': float(centrality),
                    'target_range': (float(min_val), float(max_val)),
                    'label': feature_labels.get(feature, feature)
                }
        
        return analysis
    
    def _generate_explanation_text(
        self,
        input_song: dict,
        recommended_song: dict,
        similarity_score: float,
        feature_analysis: Dict,
        common_attributes: Dict
    ) -> str:
        """Generate natural language explanation for song recommendation."""
        parts = []
        
        # Opening statement
        parts.append(
            f"Because you played \"{input_song['name']}\" by {input_song['artists']}, "
            f"we recommend \"{recommended_song['name']}\" by {recommended_song['artists']}."
        )
        
        # Similarity statement
        similarity_pct = int(similarity_score * 100)
        parts.append(f"These songs are {similarity_pct}% similar overall.")
        
        # Common attributes
        if 'artists' in common_attributes:
            parts.append(f"They share the same artist(s): {', '.join(common_attributes['artists'])}.")
        
        if 'era' in common_attributes:
            parts.append(f"Both are from the {common_attributes['era']}.")
        
        # Top similar features
        top_features = list(feature_analysis.items())[:3]
        if top_features:
            feature_descriptions = []
            for feature, data in top_features:
                if data['similarity'] > 0.8:
                    feature_descriptions.append(data['label'])
            
            if feature_descriptions:
                features_str = ', '.join(feature_descriptions)
                parts.append(f"They have similar {features_str}.")
        
        # Popularity note
        if recommended_song['popularity'] > 70:
            parts.append("This is a popular track that many users enjoy.")
        elif recommended_song['popularity'] < 30:
            parts.append("This is a hidden gem you might not have discovered yet.")
        
        return ' '.join(parts)
    
    def _generate_mood_explanation(
        self,
        song: dict,
        mood: str,
        mood_fit: float,
        mood_features: Dict
    ) -> str:
        """Generate natural language explanation for mood recommendation."""
        parts = []
        
        # Opening statement
        parts.append(
            f"\"{song['name']}\" by {song['artists']} is perfect for a {mood} mood."
        )
        
        # Mood fit score
        fit_pct = int(mood_fit * 100)
        parts.append(f"This song has a {fit_pct}% match with the {mood} mood profile.")
        
        # Feature contributions
        strong_features = [
            data['label'] for feature, data in mood_features.items()
            if data['in_range'] and data['centrality'] > 0.7
        ]
        
        if strong_features:
            features_str = ', '.join(strong_features[:2])
            parts.append(f"It has the right {features_str} for this mood.")
        
        # Mood-specific descriptions
        mood_descriptions = {
            'happy': 'uplifting and positive vibe',
            'chill': 'relaxed and mellow atmosphere',
            'sad': 'melancholic and introspective feeling',
            'energetic': 'high-energy and dynamic sound'
        }
        
        if mood in mood_descriptions:
            parts.append(f"The song has a {mood_descriptions[mood]}.")
        
        return ' '.join(parts)
    
    def _generate_hybrid_explanation(
        self,
        input_songs: List[dict],
        recommended_song: dict,
        most_similar_song: dict,
        avg_similarity: float,
        mood: Optional[str]
    ) -> str:
        """Generate natural language explanation for hybrid recommendation."""
        parts = []
        
        # Opening with multiple songs
        if len(input_songs) == 2:
            parts.append(
                f"Because you played \"{input_songs[0]['name']}\" and "
                f"\"{input_songs[1]['name']}\", "
            )
        else:
            parts.append(f"Based on your {len(input_songs)} selected songs, ")
        
        parts.append(
            f"we recommend \"{recommended_song['name']}\" by {recommended_song['artists']}."
        )
        
        # Most similar song
        parts.append(
            f"It's most similar to \"{most_similar_song['name']}\" "
            f"from your selection."
        )
        
        # Average similarity
        similarity_pct = int(avg_similarity * 100)
        parts.append(
            f"On average, it has {similarity_pct}% similarity with your selected songs."
        )
        
        # Mood filter
        if mood:
            parts.append(f"It also matches your desired {mood} mood.")
        
        # Closing
        parts.append("Users who enjoy similar combinations often love this track.")
        
        return ' '.join(parts)
    
    def _generate_hybrid_explanation(
        self,
        recommended_song: Dict,
        content_score: float,
        seq_explanation: Dict,
        context_indices: List[int]
    ) -> str:
        """Generate natural language explanation for hybrid recommendation."""
        parts = []
        
        # Sequence pattern part
        if seq_explanation.get("explanation_text"):
            parts.append(seq_explanation["explanation_text"])
        
        # Content similarity part
        if content_score > 0.7:
            if context_indices:
                context_song = self.processor.get_song_by_index(context_indices[-1])
                parts.append(
                    f"Also musically similar to '{context_song['name']}' "
                    f"({int(content_score * 100)}% match in audio features)."
                )
        elif content_score > 0.5:
            parts.append("Shares similar musical characteristics with your recent selections.")
        
        # Default if no specific patterns
        if not parts:
            parts.append(
                f"'{recommended_song['name']}' by {recommended_song['artists']} "
                f"matches your listening patterns and musical preferences."
            )
        
        return ' '.join(parts)
    
    def explain_sequence_recommendation(
        self,
        recommended_idx: int,
        context_song_indices: List[int],
        sequence_miner,  # SequentialPatternMiner
        content_score: float = 0.0,
        sequence_score: float = 0.0
    ) -> Dict:
        """
        Generate explanation for sequence-based recommendation.
        
        Args:
            recommended_idx: Index of recommended song
            context_song_indices: Indices of context songs (recent listening history)
            sequence_miner: SequentialPatternMiner with patterns
            content_score: Content-based similarity score
            sequence_score: Sequence pattern score
            
        Returns:
            Dict with combined explanation (content + sequence)
        """
        recommended_song = self.processor.get_song_by_index(recommended_idx)
        
        # Get song IDs for sequence explanation
        recommended_id = recommended_song['id']
        context_song_ids = [
            self.processor.get_song_by_index(idx)['id'] 
            for idx in context_song_indices
        ]
        
        # Get sequence pattern explanation
        seq_explanation = sequence_miner.get_pattern_explanation(
            recommended_id,
            context_song_ids
        )
        
        # Build combined explanation
        explanation = {
            "recommended_song": {
                "name": recommended_song['name'],
                "artist": recommended_song['artists'],
                "id": recommended_id
            },
            "content_score": float(content_score),
            "sequence_score": float(sequence_score),
            "combined_score": float(content_score + sequence_score) / 2,
            "sequence_pattern": seq_explanation.get("pattern"),
            "pattern_support": seq_explanation.get("support", 0.0),
            "pattern_confidence": seq_explanation.get("confidence", 0.0),
            "explanation": self._generate_hybrid_explanation(
                recommended_song,
                content_score,
                seq_explanation,
                context_song_indices
            )
        }
        
        return explanation


if __name__ == "__main__":
    from data_processor import DataProcessor
    from recommendation_engine import RecommendationEngine
    
    # Initialize
    processor = DataProcessor()
    processor.initialize()
    
    engine = RecommendationEngine(processor)
    explainer = ExplainabilityEngine(processor, engine)
    
    # Test explanation
    test_song = processor.data.iloc[100]
    recommendations = engine.song_based_recommendations(test_song['id'], n_recommendations=3)
    
    print(f"\nInput: {test_song['name']} by {test_song['artists']}\n")
    
    for idx, score in recommendations:
        explanation = explainer.explain_song_recommendation(test_song['id'], idx)
        print(f"\nRecommendation: {explanation['recommended_song']['name']}")
        print(f"Explanation: {explanation['explanation']}")
        print(f"Similarity: {explanation['similarity_score']:.2f}")
    
