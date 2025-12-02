"""Recommendation engines module."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Tuple, Optional
import random
import config
import logging
from sequence_mining import SequentialPatternMiner

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine with multiple algorithms."""
    
    def __init__(self, data_processor):
        self.processor = data_processor
        self.similarity_cache = {}
        
    def compute_similarity(self, idx1: int, idx2: int) -> float:
        """Compute cosine similarity between two songs."""
        vec1 = self.processor.get_feature_vector(idx1)
        vec2 = self.processor.get_feature_vector(idx2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return float(similarity)
    
    def compute_batch_similarity(self, idx: int, candidate_indices: List[int]) -> np.ndarray:
        """Compute similarity between one song and multiple candidates."""
        vec = self.processor.get_feature_vector(idx)
        if vec is None:
            return np.zeros(len(candidate_indices))
        
        candidate_vecs = np.array([
            self.processor.get_feature_vector(i) 
            for i in candidate_indices
        ])
        
        similarities = cosine_similarity([vec], candidate_vecs)[0]
        return similarities
    
    def song_based_recommendations(
        self, 
        song_id: str, 
        n_recommendations: int = 10,
        diversity_weight: float = 0.7
    ) -> List[Tuple[int, float]]:
        """
        Generate song-based recommendations using content-based filtering.
        
        Args:
            song_id: ID of the input song
            n_recommendations: Number of recommendations to return
            diversity_weight: Weight for diversity (0-1, higher = more diverse)
            
        Returns:
            List of (song_index, similarity_score) tuples
        """
        # Get input song
        input_idx = self.processor.song_id_to_idx.get(song_id)
        if input_idx is None:
            return []
        
        input_song = self.processor.get_song_by_index(input_idx)
        input_artists = input_song['artists_parsed']
        input_cluster = input_song['cluster']
        
        # Get candidate songs from same and nearby clusters
        candidate_indices = set()
        
        # Add songs from same cluster
        candidate_indices.update(self.processor.get_songs_in_cluster(input_cluster))
        
        # Add songs from nearby clusters (based on cluster center distance)
        cluster_centers = self.processor.kmeans_model.cluster_centers_
        input_center = cluster_centers[input_cluster]
        
        # Find closest clusters
        distances = euclidean_distances([input_center], cluster_centers)[0]
        nearby_clusters = np.argsort(distances)[1:6]  # Top 5 nearby clusters
        
        for cluster_id in nearby_clusters:
            candidate_indices.update(self.processor.get_songs_in_cluster(cluster_id))
        
        # Remove input song
        candidate_indices.discard(input_idx)
        candidate_indices = list(candidate_indices)
        
        # Compute similarities
        similarities = self.compute_batch_similarity(input_idx, candidate_indices)
        
        # Create scored candidates
        candidates = []
        for i, candidate_idx in enumerate(candidate_indices):
            similarity_score = similarities[i]
            
            # Apply diversity penalty
            candidate_song = self.processor.get_song_by_index(candidate_idx)
            candidate_artists = candidate_song['artists_parsed']
            
            # Penalty for same artist
            artist_penalty = 1.0
            if any(artist in input_artists for artist in candidate_artists):
                artist_penalty = 0.5
            
            # Boost for popularity (slightly)
            popularity_boost = 1.0 + (candidate_song['popularity'] / 1000.0)
            
            final_score = similarity_score * artist_penalty * popularity_boost
            
            if final_score >= config.MIN_SIMILARITY_THRESHOLD:
                candidates.append((candidate_idx, final_score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering
        recommendations = self._apply_diversity_filter(
            candidates, 
            n_recommendations,
            diversity_weight
        )
        
        return recommendations[:n_recommendations]
    
    def mood_based_recommendations(
        self, 
        mood: str, 
        n_recommendations: int = 10,
        include_popular: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate mood-based recommendations.
        
        Args:
            mood: Target mood (happy, chill, sad, energetic)
            n_recommendations: Number of recommendations
            include_popular: Whether to bias towards popular songs
            
        Returns:
            List of (song_index, score) tuples
        """
        if mood not in config.MOOD_CRITERIA:
            logger.warning(f"Invalid mood requested: {mood}")
            return []
        
        # Get songs matching the mood
        mood_indices = self.processor.get_songs_by_mood(mood, limit=500)
        logger.info(f"Found {len(mood_indices)} songs for mood: {mood}")
        
        if not mood_indices:
            logger.warning(f"No songs found for mood: {mood}")
            return []
        
        # Calculate mood match score based on how well song fits mood criteria
        mood_scores = []
        for idx in mood_indices:
            song = self.processor.get_song_by_index(idx)
            
            # Calculate mood fit score
            mood_score = self._calculate_mood_fit(song, mood)
            logger.debug(f"Mood score for idx {idx}: {mood_score} (type: {type(mood_score)})")
            
            # Factor in popularity if requested
            if include_popular:
                try:
                    popularity = float(song['popularity']) if song['popularity'] is not None else 0.0
                    popularity_factor = 1.0 + (popularity / 100.0)
                    mood_score *= popularity_factor
                except (ValueError, TypeError):
                    logger.warning(f"Invalid popularity value for song idx {idx}: {song.get('popularity')}")
            
            mood_scores.append((idx, mood_score))
        
        # Sort by mood score
        mood_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top candidates for diversity filtering
        top_candidates = mood_scores[:n_recommendations * 3]
        
        # Apply clustering-based diversity for better variety
        if len(top_candidates) > n_recommendations:
            diverse_recommendations = self._cluster_based_diversity(
                top_candidates,
                n_recommendations * 2
            )
        else:
            diverse_recommendations = top_candidates
        
        # Apply artist diversity filter
        diverse_recommendations = self._apply_diversity_filter(
            diverse_recommendations,
            n_recommendations,
            diversity_weight=0.8
        )
        
        return diverse_recommendations[:n_recommendations]
    
    def hybrid_recommendations(
        self,
        song_ids: List[str],
        mood: Optional[str] = None,
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations based on multiple songs and optional mood.
        
        Args:
            song_ids: List of input song IDs
            mood: Optional mood filter
            n_recommendations: Number of recommendations
            
        Returns:
            List of (song_index, score) tuples
        """
        if not song_ids:
            return []
        
        # Get recommendations from each song
        all_recommendations = {}
        
        for song_id in song_ids:
            recs = self.song_based_recommendations(song_id, n_recommendations * 3)
            for idx, score in recs:
                if idx not in all_recommendations:
                    all_recommendations[idx] = []
                all_recommendations[idx].append(score)
        
        # Aggregate scores (average)
        aggregated = []
        for idx, scores in all_recommendations.items():
            avg_score = np.mean(scores)
            count_bonus = len(scores) / len(song_ids)  # Bonus for appearing in multiple recommendations
            final_score = avg_score * (1.0 + count_bonus * 0.5)
            aggregated.append((idx, final_score))
        
        # Apply mood filter if specified
        if mood and mood in config.MOOD_CRITERIA:
            mood_indices = set(self.processor.get_songs_by_mood(mood, limit=1000))
            aggregated = [(idx, score) for idx, score in aggregated if idx in mood_indices]
        
        # Sort and return
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        return aggregated[:n_recommendations]
    
    def _calculate_mood_fit(self, song: dict, mood: str) -> float:
        """
        Calculate how well a song fits a mood using weighted feature scoring.
        
        Args:
            song: Song dictionary with audio features
            mood: Target mood
            
        Returns:
            Weighted mood fit score (0.0 to 1.0+)
        """
        criteria = config.MOOD_CRITERIA[mood]
        feature_weights = config.MOOD_FEATURE_WEIGHTS.get(mood, {})
        
        fit_score = 0.0
        total_weight = 0.0
        
        for feature, (min_val, max_val) in criteria.items():
            if feature not in song:
                continue
                
            try:
                value = float(song[feature]) if song[feature] is not None else 0.0
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for feature {feature}: {song.get(feature)}")
                continue
            
            # Get feature weight (default to 1.0 if not specified)
            weight = feature_weights.get(feature, 1.0)
            total_weight += weight
            
            # Calculate how well value fits in range
            range_size = max_val - min_val
            if range_size == 0:
                range_size = 0.01  # Avoid division by zero
            
            if min_val <= value <= max_val:
                # Value is in range - calculate position in range
                center = (min_val + max_val) / 2
                distance_from_center = abs(value - center)
                normalized_distance = distance_from_center / (range_size / 2)
                
                # Gaussian-like scoring: closer to center = higher score
                feature_score = 1.0 - (normalized_distance ** 2 * 0.5)
                fit_score += feature_score * weight
            else:
                # Value is out of range - apply penalty based on distance
                if value < min_val:
                    distance = min_val - value
                else:
                    distance = value - max_val
                
                # Exponential decay penalty (further = lower score)
                normalized_distance = min(distance / range_size, 2.0)
                penalty_score = max(0, 0.5 - (normalized_distance ** 2 * 0.25))
                fit_score += penalty_score * weight
        
        # Normalize by total weight
        return fit_score / total_weight if total_weight > 0 else 0.0
    
    def _cluster_based_diversity(
        self,
        candidates: List[Tuple[int, float]],
        n_recommendations: int
    ) -> List[Tuple[int, float]]:
        """
        Select diverse songs from different clusters to ensure sonic variety.
        Uses round-robin selection from cluster groups.
        """
        if not candidates:
            return []
        
        # Group by cluster
        cluster_groups = {}
        for idx, score in candidates:
            song = self.processor.get_song_by_index(idx)
            cluster = song.get('cluster', 0)
            
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append((idx, score))
        
        # Sort each cluster by score
        for cluster in cluster_groups:
            cluster_groups[cluster].sort(key=lambda x: x[1], reverse=True)
        
        # Round-robin selection from clusters
        selected = []
        cluster_ids = list(cluster_groups.keys())
        cluster_idx = 0
        
        while len(selected) < n_recommendations and cluster_groups:
            if not cluster_ids:
                break
                
            current_cluster = cluster_ids[cluster_idx % len(cluster_ids)]
            
            if cluster_groups[current_cluster]:
                selected.append(cluster_groups[current_cluster].pop(0))
            else:
                cluster_ids.remove(current_cluster)
            
            cluster_idx += 1
        
        return selected
    
    def _apply_diversity_filter(
        self, 
        candidates: List[Tuple[int, float]], 
        n_recommendations: int,
        diversity_weight: float = 0.7
    ) -> List[Tuple[int, float]]:
        """
        Apply diversity filtering to recommendations.
        Ensures variety in artists and clusters.
        """
        if not candidates:
            return []
        
        selected = []
        seen_artists = set()
        seen_clusters = set()
        
        # First pass: select diverse recommendations
        for idx, score in candidates:
            if len(selected) >= n_recommendations * 2:
                break
            
            song = self.processor.get_song_by_index(idx)
            artists = song['artists_parsed']
            cluster = song['cluster']
            
            # Calculate diversity penalty
            artist_penalty = 1.0
            cluster_penalty = 1.0
            
            # Penalize if artist already seen
            if any(artist in seen_artists for artist in artists):
                artist_penalty = diversity_weight
            
            # Penalize if cluster already seen
            if cluster in seen_clusters:
                cluster_penalty = diversity_weight
            
            adjusted_score = score * artist_penalty * cluster_penalty
            selected.append((idx, adjusted_score))
            
            # Update seen sets
            seen_artists.update(artists)
            seen_clusters.add(cluster)
        
        # Sort by adjusted score
        selected.sort(key=lambda x: x[1], reverse=True)
        
        return selected
    
    def get_similar_songs_in_cluster(self, song_id: str, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Find similar songs within the same cluster."""
        input_idx = self.processor.song_id_to_idx.get(song_id)
        if input_idx is None:
            return []
        
        input_song = self.processor.get_song_by_index(input_idx)
        cluster_id = input_song['cluster']
        
        # Get all songs in cluster
        cluster_songs = self.processor.get_songs_in_cluster(cluster_id)
        cluster_songs = [idx for idx in cluster_songs if idx != input_idx]
        
        if not cluster_songs:
            return []
        
        # Compute similarities
        similarities = self.compute_batch_similarity(input_idx, cluster_songs)
        
        # Create results
        results = [(cluster_songs[i], similarities[i]) for i in range(len(cluster_songs))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:n_recommendations]
    
    def sequence_aware_recommendations(
        self,
        current_songs: List[int],  # Song indices
        sequence_miner: Optional[SequentialPatternMiner] = None,
        num_recommendations: int = 10,
        sequence_weight: float = 0.3  # Weight for sequence vs content-based
    ) -> List[Tuple[int, float]]:
        """
        Hybrid recommendations combining content-based and sequential pattern mining.
        
        Args:
            current_songs: List of song indices (recent listening history)
            sequence_miner: SequentialPatternMiner instance with patterns
            num_recommendations: Number of recommendations to return
            sequence_weight: Weight for sequence score (0-1), content gets (1-weight)
            
        Returns:
            List of (song_index, combined_score) tuples
        """
        if not current_songs:
            logger.warning("No current songs provided for sequence-aware recommendations")
            return []
        
        # Convert indices to song IDs for hybrid method
        current_song_ids = [
            self.processor.get_song_by_index(idx)['id'] 
            for idx in current_songs
        ]
        
        # Get content-based recommendations using hybrid method
        content_recs = self.hybrid_recommendations(
            current_song_ids,
            n_recommendations=num_recommendations * 3  # Get more for merging
        )
        
        # If no sequence miner provided, return pure content-based
        if not sequence_miner or not sequence_miner.patterns:
            logger.info("No sequence patterns available, using pure content-based recommendations")
            return content_recs[:num_recommendations]
        
        # Get sequence-based predictions (current_song_ids already computed above)
        sequence_predictions = sequence_miner.predict_next_songs(
            current_song_ids,
            top_k=num_recommendations * 2
        )
        
        # Convert sequence predictions to indices and scores
        sequence_recs_dict = {}
        for song_id, seq_score in sequence_predictions:
            song = self.processor.get_song_by_id(song_id)
            if song is not None and 'index' in song:
                idx = song['index']
                # Don't recommend songs already in current_songs
                if idx not in current_songs:
                    sequence_recs_dict[idx] = seq_score
        
        # Build content-based dict for easier merging
        content_recs_dict = {idx: score for idx, score in content_recs}
        
        # Merge recommendations with weighted scoring
        merged_scores = {}
        all_indices = set(content_recs_dict.keys()) | set(sequence_recs_dict.keys())
        
        for idx in all_indices:
            content_score = content_recs_dict.get(idx, 0.0)
            sequence_score = sequence_recs_dict.get(idx, 0.0)
            
            # Normalize scores (they may be in different ranges)
            # Content scores are typically 0-1, sequence scores are 0-1
            combined_score = (1 - sequence_weight) * content_score + sequence_weight * sequence_score
            
            # Bonus if song appears in both recommendation types
            if idx in content_recs_dict and idx in sequence_recs_dict:
                combined_score *= 1.2  # 20% boost for consensus
            
            merged_scores[idx] = combined_score
        
        # Sort by combined score and return top-k
        ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:num_recommendations]


if __name__ == "__main__":
    from data_processor import DataProcessor
    
    # Initialize
    processor = DataProcessor()
    processor.initialize()
    
    engine = RecommendationEngine(processor)
    
    # Test song-based recommendations
    print("\nTesting song-based recommendations...")
    test_song = processor.data.iloc[0]
    print(f"Input song: {test_song['name']} by {test_song['artists']}")
    
    recommendations = engine.song_based_recommendations(test_song['id'], n_recommendations=5)
    print("\nRecommendations:")
    for idx, score in recommendations:
        song = processor.get_song_by_index(idx)
        print(f"  {song['name']} by {song['artists']} (score: {score:.3f})")
    
    # Test mood-based recommendations
    print("\n\nTesting mood-based recommendations (happy)...")
    mood_recs = engine.mood_based_recommendations('happy', n_recommendations=5)
    for idx, score in mood_recs:
        song = processor.get_song_by_index(idx)
        print(f"  {song['name']} by {song['artists']} (score: {score:.3f})")
