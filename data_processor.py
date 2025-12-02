"""Data processing and feature engineering module."""

import os
# Set PyArrow environment variable before any imports
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
from typing import Dict, List, Tuple, Optional
import config

# Conditional PySpark imports
if config.USE_PYSPARK:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StringType, DoubleType
    from pyspark.ml.feature import StandardScaler as SparkScaler, VectorAssembler
    from pyspark.ml.clustering import KMeans as SparkKMeans
    from pyspark.ml.linalg import Vectors, VectorUDT
    import pyspark.pandas as ps


# Define UDF functions at module level to avoid serialization issues
def parse_artists_str(artist_str: str) -> List[str]:
    """Parse artist string to list - used as PySpark UDF."""
    if not artist_str:
        return []
    try:
        artist_str = artist_str.strip("[]'\"")
        artists = [a.strip().strip("'\"") for a in artist_str.split("', '")]
        return artists
    except:
        return [artist_str]


class PySparkDataProcessor:
    """PySpark-based data processor for handling large-scale data."""
    
    def __init__(self):
        self.spark = None
        self.data = None
        self.data_pandas = None  # Keep a pandas version for compatibility
        self.data_by_artist = None
        self.data_by_genres = None
        self.data_by_year = None
        self.data_with_genres = None
        
        self.scaler = None
        self.scaler_model = None
        self.normalized_features = None
        self.kmeans_model = None
        
        # Indexes for fast lookup
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.mood_to_songs = {mood: [] for mood in config.MOOD_CRITERIA.keys()}
        self.genre_to_songs = {}
        self.artist_to_songs = {}
        self.cluster_to_songs = {}
        
        self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session with reduced logging."""
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Set environment variables to reduce warnings
        os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
        
        # Configure logging
        import logging
        logging.getLogger('py4j').setLevel(logging.ERROR)
        
        builder = SparkSession.builder
        for key, value in config.PYSPARK_CONFIG.items():
            builder = builder.config(key, value)
        
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")  # Only show errors
        print(f"✓ Spark session initialized: {self.spark.version}")
    
    def load_data(self):
        """Load all CSV files using PySpark."""
        print("Loading datasets with PySpark...")
        # Read without inferSchema to avoid type inference issues with mixed data
        # We'll handle type conversion manually in _clean_data()
        self.data = self.spark.read.csv(config.DATA_FILES["main"], header=True, inferSchema=False)
        self.data_by_artist = self.spark.read.csv(config.DATA_FILES["by_artist"], header=True, inferSchema=False)
        self.data_by_genres = self.spark.read.csv(config.DATA_FILES["by_genres"], header=True, inferSchema=False)
        self.data_by_year = self.spark.read.csv(config.DATA_FILES["by_year"], header=True, inferSchema=False)
        self.data_with_genres = self.spark.read.csv(config.DATA_FILES["with_genres"], header=True, inferSchema=False)
        
        # Clean data
        self._clean_data()
        print(f"✓ Loaded {self.data.count()} songs")
    
    def _clean_data(self):
        """Clean and prepare data using PySpark."""
        # Convert ID to string to avoid integer ID conflicts (0, 1, etc.)
        self.data = self.data.withColumn('id', F.col('id').cast(StringType()))
        
        # Drop duplicates
        self.data = self.data.dropDuplicates(['id'])
        
        # Cast audio feature columns to double type, handling invalid values
        # Some CSVs may have corrupt data with text in numeric columns
        for feature in config.AUDIO_FEATURES:
            if feature in self.data.columns:
                # Use regexp to validate numeric format, then cast (similar to pd.to_numeric with errors='coerce')
                # Match: optional negative, digits, optional decimal, optional scientific notation (e.g., 1e-05)
                self.data = self.data.withColumn(
                    feature,
                    F.when(
                        F.col(feature).cast(StringType()).rlike(r'^\s*-?\d+\.?\d*([eE][+-]?\d+)?\s*$'),
                        F.col(feature).cast(DoubleType())
                    ).otherwise(None)
                )
        
        # Handle missing values - drop rows with nulls in required columns
        required_cols = ['id', 'name'] + config.AUDIO_FEATURES
        self.data = self.data.dropna(subset=required_cols)
        
        # Parse artists column using module-level function
        parse_artists_udf = F.udf(parse_artists_str, ArrayType(StringType()))
        self.data = self.data.withColumn('artists_parsed', parse_artists_udf(F.col('artists')))
        
        # Cache the data for faster access
        self.data = self.data.cache()
        
        # Create a pandas version for compatibility
        self.data_pandas = self.data.toPandas().reset_index(drop=True)
    
    def create_indexes(self):
        """Create lookup indexes for fast retrieval."""
        print("Creating indexes...")
        
        # Song ID to index mapping
        for idx, row in enumerate(self.data_pandas.itertuples()):
            song_id = row.id
            self.song_id_to_idx[song_id] = idx
            self.idx_to_song_id[idx] = song_id
        
        # Artist to songs mapping
        for idx, row in self.data_pandas.iterrows():
            artists = row['artists_parsed']
            if isinstance(artists, list):
                for artist in artists:
                    if artist not in self.artist_to_songs:
                        self.artist_to_songs[artist] = []
                    self.artist_to_songs[artist].append(idx)
        
        print(f"✓ Indexed {len(self.song_id_to_idx)} songs, {len(self.artist_to_songs)} artists")
    
    def extract_mood_features(self):
        """Classify songs into mood categories using PySpark."""
        print("Extracting mood features...")
        
        for mood, criteria in config.MOOD_CRITERIA.items():
            # Build filter condition
            condition = None
            for feature, (min_val, max_val) in criteria.items():
                if feature in self.data.columns:
                    feature_condition = (F.col(feature) >= min_val) & (F.col(feature) <= max_val)
                    condition = feature_condition if condition is None else condition & feature_condition
            
            if condition is not None:
                mood_df = self.data.filter(condition)
                # Get indices from pandas dataframe
                mood_ids = [row.id for row in mood_df.select('id').collect()]
                mood_indices = [self.song_id_to_idx[sid] for sid in mood_ids if sid in self.song_id_to_idx]
                self.mood_to_songs[mood] = mood_indices
                
                print(f"  ✓ {mood}: {len(mood_indices)} songs")
    
    def normalize_features(self):
        """Normalize audio features using PySpark ML."""
        print("Normalizing features...")
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=config.AUDIO_FEATURES,
            outputCol="features"
        )
        
        data_with_features = assembler.transform(self.data)
        
        # Normalize using PySpark StandardScaler
        scaler = SparkScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        self.scaler_model = scaler.fit(data_with_features)
        scaled_data = self.scaler_model.transform(data_with_features)
        
        # Extract normalized features as numpy array for compatibility
        scaled_features_list = scaled_data.select("scaled_features").rdd.map(lambda row: row[0].toArray()).collect()
        self.normalized_features = np.array(scaled_features_list)
        
        # Also keep a sklearn scaler for compatibility
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        feature_matrix = self.data_pandas[config.AUDIO_FEATURES].values
        self.scaler.fit(feature_matrix)
        
        print(f"✓ Normalized {len(config.AUDIO_FEATURES)} features for {len(self.normalized_features)} songs")
    
    def cluster_songs(self):
        """Perform K-Means clustering using PySpark ML."""
        print(f"Clustering songs into {config.N_CLUSTERS} clusters...")
        
        # Create DataFrame with features for clustering
        features_rdd = self.spark.sparkContext.parallelize(
            [(i, Vectors.dense(feat)) for i, feat in enumerate(self.normalized_features)]
        )
        features_df = self.spark.createDataFrame(features_rdd, ["id", "features"])
        
        # Perform K-Means clustering
        kmeans = SparkKMeans(
            k=config.N_CLUSTERS,
            seed=config.RANDOM_STATE,
            featuresCol="features",
            predictionCol="cluster"
        )
        
        self.kmeans_model = kmeans.fit(features_df)
        clustered_data = self.kmeans_model.transform(features_df)
        
        # Extract cluster labels
        cluster_labels = [row.cluster for row in clustered_data.orderBy("id").collect()]
        
        # Add cluster labels to pandas dataframe
        self.data_pandas['cluster'] = cluster_labels
        
        # Create cluster to songs mapping
        for cluster_id in range(config.N_CLUSTERS):
            cluster_indices = self.data_pandas[self.data_pandas['cluster'] == cluster_id].index.tolist()
            self.cluster_to_songs[cluster_id] = cluster_indices
        
        print(f"✓ Clustering complete. Average cluster size: {len(self.data_pandas) / config.N_CLUSTERS:.1f}")
    
    def get_song_by_id(self, song_id: str) -> Optional[pd.Series]:
        """Get song data by ID."""
        if song_id in self.song_id_to_idx:
            idx = self.song_id_to_idx[song_id]
            return self.data_pandas.iloc[idx]
        return None
    
    def get_song_by_index(self, idx: int) -> Optional[pd.Series]:
        """Get song data by index."""
        if 0 <= idx < len(self.data_pandas):
            return self.data_pandas.iloc[idx]
        return None
    
    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search songs by name or artist using PySpark."""
        query_lower = query.lower()
        
        # Search using PySpark
        matches = self.data.filter(
            F.lower(F.col("name")).contains(query_lower) | 
            F.lower(F.col("artists")).contains(query_lower)
        ).limit(limit)
        
        results = []
        for row in matches.collect():
            results.append({
                'id': row.id,
                'name': row.name,
                'artists': row.artists,
                'year': int(row.year),
                'popularity': int(row.popularity)
            })
        
        return results
    
    def get_songs_by_mood(self, mood: str, limit: int = 50) -> List[int]:
        """Get song indices for a specific mood."""
        if mood in self.mood_to_songs:
            indices = self.mood_to_songs[mood]
            if indices:
                mood_df = self.data_pandas.iloc[indices].copy()
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
        
        # Extract cluster centers from PySpark model and create sklearn model for compatibility
        cluster_centers = self.kmeans_model.clusterCenters()
        sklearn_kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE)
        sklearn_kmeans.cluster_centers_ = np.array(cluster_centers)
        sklearn_kmeans._n_threads = 1
        
        data_dict = {
            'data': self.data_pandas,
            'scaler': self.scaler,
            'normalized_features': self.normalized_features,
            'kmeans_model': sklearn_kmeans,  # Save sklearn version for compatibility
            'song_id_to_idx': self.song_id_to_idx,
            'idx_to_song_id': self.idx_to_song_id,
            'mood_to_songs': self.mood_to_songs,
            'artist_to_songs': self.artist_to_songs,
            'cluster_to_songs': self.cluster_to_songs,
            'use_pyspark': True
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print("✓ Data saved successfully")
    
    def load_processed_data(self, filepath: str = "processed_data.pkl"):
        """Load previously processed data."""
        print(f"Loading processed data from {filepath}...")
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data_pandas = data_dict['data']
        self.scaler = data_dict['scaler']
        self.normalized_features = data_dict['normalized_features']
        self.kmeans_model = data_dict['kmeans_model']
        self.song_id_to_idx = data_dict['song_id_to_idx']
        self.idx_to_song_id = data_dict['idx_to_song_id']
        self.mood_to_songs = data_dict['mood_to_songs']
        self.artist_to_songs = data_dict['artist_to_songs']
        self.cluster_to_songs = data_dict['cluster_to_songs']
        
        # Recreate PySpark DataFrame from pandas for search operations
        self.data = self.spark.createDataFrame(self.data_pandas)
        
        print("✓ Data loaded successfully")
    
    def initialize(self, force_reload: bool = False):
        """Initialize the data processor."""
        processed_file = "processed_data_pyspark.pkl"
        
        if not force_reload and os.path.exists(processed_file):
            self.load_processed_data(processed_file)
        else:
            self.load_data()
            self.create_indexes()
            self.normalize_features()
            self.extract_mood_features()
            self.cluster_songs()
            self.save_processed_data(processed_file)
        
        print("PySpark Data processor initialized")
    
    def stop_spark(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            print("✓ Spark session stopped")


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
        # Convert ID to string to ensure consistency
        self.data['id'] = self.data['id'].astype(str)
        
        # Drop duplicates
        self.data = self.data.drop_duplicates(subset=['id'])
        
        # Cast audio feature columns to numeric, replacing invalid values with NaN
        for feature in config.AUDIO_FEATURES:
            if feature in self.data.columns:
                self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce')
        
        # Handle missing values - drop rows with NaN in required columns
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


def create_data_processor():
    """
    Factory function to create the appropriate data processor based on config.
    
    Returns:
        DataProcessor or PySparkDataProcessor based on config.USE_PYSPARK
    """
    if config.USE_PYSPARK:
        print("Using PySpark for data processing")
        return PySparkDataProcessor()
    else:
        print("Using Pandas for data processing")
        return DataProcessor()


if __name__ == "__main__":
    # Test the data processor
    processor = create_data_processor()
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
    
    # Clean up Spark session if using PySpark
    if config.USE_PYSPARK and hasattr(processor, 'stop_spark'):
        processor.stop_spark()
