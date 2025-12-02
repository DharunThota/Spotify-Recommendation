"""Configuration settings for the recommendation system."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
# Each mood now includes weighted features for better matching
MOOD_CRITERIA = {
    "happy": {
        "valence": (0.6, 1.0),
        "energy": (0.5, 1.0),
        "danceability": (0.5, 1.0),
        "tempo": (100, 180),
        "acousticness": (0.0, 0.5),  # Less acoustic = more upbeat
        "speechiness": (0.0, 0.3)     # Less speech = more musical
    },
    "chill": {
        "energy": (0.0, 0.5),
        "tempo": (60, 120),
        "acousticness": (0.4, 1.0),
        "valence": (0.3, 0.7),        # Neutral to slightly positive
        "instrumentalness": (0.3, 1.0), # More instrumental
        "loudness": (-20, -8)          # Quieter songs
    },
    "sad": {
        "valence": (0.0, 0.4),
        "energy": (0.0, 0.5),
        "acousticness": (0.3, 1.0),
        "tempo": (60, 100),            # Slower tempo
        "instrumentalness": (0.0, 0.7),
        "loudness": (-20, -6)
    },
    "energetic": {
        "energy": (0.7, 1.0),
        "tempo": (120, 200),
        "danceability": (0.6, 1.0),
        "valence": (0.4, 1.0),
        "loudness": (-8, 0),            # Louder songs
        "speechiness": (0.0, 0.3)
    },
    "romantic": {
        "valence": (0.4, 0.8),
        "energy": (0.2, 0.6),
        "acousticness": (0.3, 0.9),
        "tempo": (70, 120),
        "instrumentalness": (0.0, 0.4),
        "danceability": (0.3, 0.7)
    },
    "focus": {
        "energy": (0.3, 0.7),
        "instrumentalness": (0.5, 1.0),  # Highly instrumental
        "speechiness": (0.0, 0.2),       # Minimal vocals
        "tempo": (80, 130),
        "valence": (0.3, 0.7),
        "acousticness": (0.2, 0.8)
    },
    "party": {
        "energy": (0.7, 1.0),
        "danceability": (0.7, 1.0),
        "valence": (0.6, 1.0),
        "tempo": (120, 180),
        "loudness": (-6, 0),
        "acousticness": (0.0, 0.3)
    },
    "melancholic": {
        "valence": (0.1, 0.4),
        "energy": (0.2, 0.5),
        "acousticness": (0.4, 1.0),
        "tempo": (60, 100),
        "instrumentalness": (0.2, 0.8),
        "loudness": (-18, -8)
    }
}

# Feature weights for mood matching (higher = more important)
MOOD_FEATURE_WEIGHTS = {
    "happy": {"valence": 2.0, "energy": 1.5, "danceability": 1.3},
    "chill": {"energy": 2.0, "acousticness": 1.5, "tempo": 1.3},
    "sad": {"valence": 2.0, "energy": 1.5, "acousticness": 1.2},
    "energetic": {"energy": 2.0, "tempo": 1.5, "danceability": 1.3},
    "romantic": {"valence": 1.5, "acousticness": 1.3, "energy": 1.2},
    "focus": {"instrumentalness": 2.0, "speechiness": 1.8, "energy": 1.2},
    "party": {"danceability": 2.0, "energy": 1.8, "valence": 1.5},
    "melancholic": {"valence": 2.0, "energy": 1.5, "acousticness": 1.3}
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
USE_PYSPARK = True  # Toggle between PySpark (True) and Pandas (False)

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

# Sequential Pattern Mining Configuration
SEQUENCE_MIN_SUPPORT = 0.3  # Minimum pattern frequency (0-1)
SEQUENCE_MAX_GAP = 2  # Maximum gap between items in sequence
SEQUENCE_SESSION_GAP_MINUTES = 30  # Time gap to consider new session
SEQUENCE_WEIGHT = 0.3  # Weight in hybrid scoring (0-1)
SEQUENCE_CACHE_FILE = os.path.join(DATA_DIR, "processed_sequences.pkl")

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Spotify Recommendation System"
API_VERSION = "1.0.0"
