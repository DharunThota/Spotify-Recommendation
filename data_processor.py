"""Data processing and feature engineering module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
from typing import Dict, List, Tuple, Optional
import config


class DataProcessor:
    """Handles all data loading, preprocessing, and feature engineering."""
    
    def __init__(self):
        self.data = None
        self.data_by_artist = None
        self.data_by_genres = None
        self.data_by_year = None
        self.data_with_genres = None
        
        self.scaler = StandardScaler()
        self.normalized_features = None
        self.kmeans_model = None
        
        # Indexes for fast lookup
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.mood_to_songs = {mood: [] for mood in config.MOOD_CRITERIA.keys()}
        self.genre_to_songs = {}
        self.artist_to_songs = {}
        self.cluster_to_songs = {}
        
    def load_data(self):
        """Load all CSV files."""
        print("Loading datasets...")
        self.data = pd.read_csv(config.DATA_FILES["main"])
        self.data_by_artist = pd.read_csv(config.DATA_FILES["by_artist"])
        self.data_by_genres = pd.read_csv(config.DATA_FILES["by_genres"])
        self.data_by_year = pd.read_csv(config.DATA_FILES["by_year"])
        self.data_with_genres = pd.read_csv(config.DATA_FILES["with_genres"])
        
        # Clean data
        self._clean_data()
        print(f"Loaded {len(self.data)} songs")
        
    def _clean_data(self):
        """Clean and prepare data."""
        # Drop duplicates
        self.data = self.data.drop_duplicates(subset=['id'])
        
        # Handle missing values
        self.data = self.data.dropna(subset=['id', 'name'] + config.AUDIO_FEATURES)
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Parse artists if it's a string representation of list
        if isinstance(self.data['artists'].iloc[0], str):
            self.data['artists_parsed'] = self.data['artists'].apply(self._parse_artists)
        else:
            self.data['artists_parsed'] = self.data['artists']
            
    def _parse_artists(self, artist_str: str) -> List[str]:
        """Parse artist string to list."""
        try:
            # Remove brackets and quotes, split by comma
            artist_str = artist_str.strip("[]'\"")
            artists = [a.strip().strip("'\"") for a in artist_str.split("', '")]
            return artists
        except:
            return [artist_str]
    
    def create_indexes(self):
        """Create lookup indexes for fast retrieval."""
        print("Creating indexes...")
        
        # Song ID to index mapping
        for idx, song_id in enumerate(self.data['id']):
            self.song_id_to_idx[song_id] = idx
            self.idx_to_song_id[idx] = song_id
        
        # Artist to songs mapping
        for idx, row in self.data.iterrows():
            artists = row['artists_parsed']
            for artist in artists:
                if artist not in self.artist_to_songs:
                    self.artist_to_songs[artist] = []
                self.artist_to_songs[artist].append(idx)
        
        print(f"Indexed {len(self.song_id_to_idx)} songs, {len(self.artist_to_songs)} artists")
    
    def extract_mood_features(self):
        """Classify songs into mood categories."""
        print("Extracting mood features...")
        
        for mood, criteria in config.MOOD_CRITERIA.items():
            mask = pd.Series([True] * len(self.data))
            
            for feature, (min_val, max_val) in criteria.items():
                if feature in self.data.columns:
                    mask &= (self.data[feature] >= min_val) & (self.data[feature] <= max_val)
            
            mood_indices = self.data[mask].index.tolist()
            self.mood_to_songs[mood] = mood_indices
            
            print(f"  {mood}: {len(mood_indices)} songs")
    
    def normalize_features(self):
        """Normalize audio features for similarity computation."""
        print("Normalizing features...")
        
        # Extract feature matrix
        feature_matrix = self.data[config.AUDIO_FEATURES].values
        
        # Normalize
        self.normalized_features = self.scaler.fit_transform(feature_matrix)
        
        print(f"Normalized {self.normalized_features.shape[1]} features for {self.normalized_features.shape[0]} songs")
    
    def cluster_songs(self):
        """Perform K-Means clustering on songs."""
        print(f"Clustering songs into {config.N_CLUSTERS} clusters...")
        
        self.kmeans_model = KMeans(
            n_clusters=config.N_CLUSTERS,
            random_state=config.RANDOM_STATE,
            n_init=10
        )
        
        cluster_labels = self.kmeans_model.fit_predict(self.normalized_features)
        self.data['cluster'] = cluster_labels
        
        # Create cluster to songs mapping
        for cluster_id in range(config.N_CLUSTERS):
            cluster_indices = self.data[self.data['cluster'] == cluster_id].index.tolist()
            self.cluster_to_songs[cluster_id] = cluster_indices
        
        print(f"Clustering complete. Average cluster size: {len(self.data) / config.N_CLUSTERS:.1f}")
    
    def get_song_by_id(self, song_id: str) -> Optional[pd.Series]:
        """Get song data by ID."""
        if song_id in self.song_id_to_idx:
            idx = self.song_id_to_idx[song_id]
            return self.data.iloc[idx]
        return None
    
    def get_song_by_index(self, idx: int) -> Optional[pd.Series]:
        """Get song data by index."""
        if 0 <= idx < len(self.data):
            return self.data.iloc[idx]
        return None
    
    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search songs by name or artist."""
        query_lower = query.lower()
        
        # Search in song names
        name_matches = self.data[
            self.data['name'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # Search in artists
        artist_matches = self.data[
            self.data['artists'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # Combine and remove duplicates
        matches = pd.concat([name_matches, artist_matches]).drop_duplicates(subset=['id'])
        matches = matches.head(limit)
        
        results = []
        for _, row in matches.iterrows():
            results.append({
                'id': row['id'],
                'name': row['name'],
                'artists': row['artists'],
                'year': int(row['year']),
                'popularity': int(row['popularity'])
            })
        
        return results
    
    def get_songs_by_mood(self, mood: str, limit: int = 50) -> List[int]:
        """Get song indices for a specific mood."""
        if mood in self.mood_to_songs:
            indices = self.mood_to_songs[mood]
            # Sort by popularity and return top N
            if indices:
                mood_df = self.data.iloc[indices].copy()
                mood_df = mood_df.sort_values('popularity', ascending=False)
                return mood_df.head(limit).index.tolist()
        return []
    
    def get_feature_vector(self, idx: int) -> np.ndarray:
        """Get normalized feature vector for a song."""
        if 0 <= idx < len(self.normalized_features):
            return self.normalized_features[idx]
        return None
    
    def get_songs_in_cluster(self, cluster_id: int) -> List[int]:
        """Get all song indices in a cluster."""
        return self.cluster_to_songs.get(cluster_id, [])
    
    def save_processed_data(self, filepath: str = "processed_data.pkl"):
        """Save processed data and models."""
        print(f"Saving processed data to {filepath}...")
        
        data_dict = {
            'data': self.data,
            'scaler': self.scaler,
            'normalized_features': self.normalized_features,
            'kmeans_model': self.kmeans_model,
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'mood_to_songs': self.mood_to_songs,
            'artist_to_songs': self.artist_to_songs,
            'cluster_to_songs': self.cluster_to_songs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print("Data saved successfully")
    
    def load_processed_data(self, filepath: str = "processed_data.pkl"):
        """Load previously processed data."""
        print(f"Loading processed data from {filepath}...")
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']
        self.scaler = data_dict['scaler']
        self.normalized_features = data_dict['normalized_features']
        self.kmeans_model = data_dict['kmeans_model']
        self.song_id_to_idx = data_dict['song_id_to_idx']
        self.idx_to_song_id = data_dict['idx_to_song_id']
        self.mood_to_songs = data_dict['mood_to_songs']
        self.artist_to_songs = data_dict['artist_to_songs']
        self.cluster_to_songs = data_dict['cluster_to_songs']
        
        print("Data loaded successfully")
    
    def initialize(self, force_reload: bool = False):
        """Initialize the data processor."""
        processed_file = "processed_data.pkl"
        
        if not force_reload and os.path.exists(processed_file):
            self.load_processed_data(processed_file)
        else:
            self.load_data()
            self.create_indexes()
            self.normalize_features()
            self.extract_mood_features()
            self.cluster_songs()
            self.save_processed_data(processed_file)
        
        print("Data processor initialized")


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    processor.initialize(force_reload=True)
    
    # Test search
    print("\nTesting search for 'love':")
    results = processor.search_songs("love", limit=5)
    for r in results:
        print(f"  {r['name']} by {r['artists']}")
    
    # Test mood extraction
    print("\nMood statistics:")
    for mood in config.MOOD_CRITERIA.keys():
        songs = processor.get_songs_by_mood(mood, limit=100)
        print(f"  {mood}: {len(songs)} songs")
