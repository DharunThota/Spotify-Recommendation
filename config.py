"""Configuration settings for the recommendation system."""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data file paths
DATA_FILES = {
    "main": os.path.join(DATA_DIR, "data.csv"),
    "by_artist": os.path.join(DATA_DIR, "data_by_artist.csv"),
    "by_genres": os.path.join(DATA_DIR, "data_by_genres.csv"),
    "by_year": os.path.join(DATA_DIR, "data_by_year.csv"),
    "with_genres": os.path.join(DATA_DIR, "data_w_genres.csv")
}

# Feature columns for similarity computation
AUDIO_FEATURES = [
    "valence", "acousticness", "danceability", "duration_ms",
    "energy", "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo"
]

# Mood definitions based on audio features
MOOD_CRITERIA = {
    "happy": {
        "valence": (0.6, 1.0),
        "energy": (0.5, 1.0),
        "danceability": (0.5, 1.0)
    },
    "chill": {
        "energy": (0.0, 0.5),
        "tempo": (0, 100),
        "acousticness": (0.4, 1.0)
    },
    "sad": {
        "valence": (0.0, 0.4),
        "energy": (0.0, 0.5),
        "acousticness": (0.3, 1.0)
    },
    "energetic": {
        "energy": (0.7, 1.0),
        "tempo": (120, 250),
        "danceability": (0.6, 1.0)
    }
}

# Spotify API Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "your_client_id_here")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "your_client_secret_here")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/callback")

# Scopes required for user data access
SPOTIFY_SCOPES = [
    "user-read-recently-played",
    "user-top-read",
    "user-library-read",
    "playlist-read-private",
    "user-read-private"
]

# Processing Engine Configuration
# Set to True to use PySpark for data processing, False to use Pandas
USE_PYSPARK = False  # Toggle between PySpark (True) and Pandas (False)

# PySpark Configuration (only used when USE_PYSPARK = True)
PYSPARK_CONFIG = {
    "spark.app.name": "SpotifyRecommendation",
    "spark.master": "local[*]",  # Use all available cores
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.local.dir": "/tmp/spark-temp",
    "spark.ui.showConsoleProgress": "false",  # Disable progress bar in console
    "spark.sql.adaptive.enabled": "true",  # Enable adaptive query execution
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

# Recommendation parameters
N_RECOMMENDATIONS = 10
MIN_SIMILARITY_THRESHOLD = 0.3
DIVERSITY_FACTOR = 0.7  # Balance between similarity and diversity

# Clustering parameters
N_CLUSTERS = 75
RANDOM_STATE = 42

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Spotify Recommendation System"
API_VERSION = "1.0.0"
