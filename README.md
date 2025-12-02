# Spotify Recommendation System

A comprehensive music recommendation system using big data algorithms, featuring song-based, mood-based, and hybrid recommendations with explainable AI.

## Features

✅ **Song-Based Recommendations**: Find similar songs based on audio features using cosine similarity and K-Means clustering

✅ **Mood-Based Recommendations**: Discover songs matching your mood (happy, chill, sad, energetic) based on audio characteristics

✅ **Hybrid Recommendations**: Combine multiple songs with optional mood filtering for personalized playlists

✅ **Explainable AI**: Every recommendation comes with human-readable explanations showing why songs were suggested

✅ **Big Data Algorithms**:
- K-Means Clustering (75 clusters)
- Cosine Similarity for content-based filtering
- Feature engineering and normalization
- Diversity filtering for varied recommendations

## Project Structure

```
ASBD/
├── data/                          # Dataset files
│   ├── data.csv                   # Main song dataset (170K+ songs)
│   ├── data_by_artist.csv         # Artist aggregated data
│   ├── data_by_genres.csv         # Genre aggregated data
│   ├── data_by_year.csv           # Year aggregated data
│   └── data_w_genres.csv          # Songs with genre information
├── static/                        # Frontend files
│   ├── index.html                 # Main UI
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend logic
├── config.py                      # Configuration settings
├── data_processor.py              # Data loading and preprocessing
├── recommendation_engine.py       # Recommendation algorithms
├── explainability.py              # Explanation generation
├── main.py                        # FastAPI backend
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify data files are in the `data/` folder**

## Usage

### 1. Initialize the System (First Run)

Process and prepare the dataset:

```bash
python data_processor.py
```

This will:
- Load all CSV files
- Normalize audio features
- Extract mood classifications
- Perform K-Means clustering
- Create indexed lookup structures
- Save processed data to `processed_data.pkl`

### 2. Start the FastAPI Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Search Songs
```
GET /api/search?query={search_term}&limit={number}
```

### Song-Based Recommendations
```
POST /api/recommend/song
Body: {
  "song_id": "string",
  "n_recommendations": 10,
  "diversity_weight": 0.7
}
```

### Mood-Based Recommendations
```
POST /api/recommend/mood
Body: {
  "mood": "happy|chill|sad|energetic",
  "n_recommendations": 10,
  "include_popular": true
}
```

### Hybrid Recommendations
```
POST /api/recommend/hybrid
Body: {
  "song_ids": ["id1", "id2", ...],
  "mood": "happy|chill|sad|energetic" (optional),
  "n_recommendations": 10
}
```

## How It Works

### 1. Data Processing
- Loads 170K+ songs with audio features (valence, energy, danceability, etc.)
- Normalizes features using StandardScaler
- Creates indexed lookup structures for fast retrieval

### 2. Mood Classification
Songs are classified into moods based on audio features:
- **Happy**: High valence + moderate-high energy
- **Chill**: Low energy + low tempo + high acousticness
- **Sad**: Low valence + low energy
- **Energetic**: High energy + high tempo + high danceability

### 3. Clustering
- K-Means clustering groups similar songs
- 75 clusters for diverse groupings
- Used for efficient candidate selection

### 4. Similarity Computation
- Cosine similarity between feature vectors
- Considers 10 audio features
- Applies diversity penalties to avoid repetition

### 5. Explainability
Generates explanations by:
- Analyzing feature similarities
- Finding common attributes (artists, era, mode)
- Calculating mood fit scores
- Creating natural language descriptions

## Audio Features Used

- **Valence**: Musical positiveness (0-1)
- **Energy**: Intensity and activity (0-1)
- **Danceability**: How suitable for dancing (0-1)
- **Acousticness**: Confidence of acoustic sound (0-1)
- **Speechiness**: Presence of spoken words (0-1)
- **Instrumentalness**: Predicts if track has no vocals (0-1)
- **Liveness**: Presence of audience (0-1)
- **Loudness**: Overall loudness in dB
- **Tempo**: Beats per minute (BPM)
- **Duration**: Track length in milliseconds

## Technical Details

### Algorithms
- **K-Means Clustering**: Groups songs into 75 clusters based on audio features
- **Cosine Similarity**: Measures similarity between song feature vectors
- **Content-Based Filtering**: Recommends based on song characteristics
- **Diversity Filtering**: Ensures varied recommendations across artists and clusters

### Performance Optimizations
- Pre-computed normalized feature vectors
- Indexed lookup structures (O(1) access)
- Batch similarity computation
- Cluster-based candidate selection
- Persistent storage of processed data

### Libraries Used
- **FastAPI**: Modern web framework for APIs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Uvicorn**: ASGI server

## Configuration

Edit `config.py` to customize:
- Number of recommendations
- Clustering parameters
- Mood criteria thresholds
- Similarity thresholds
- Diversity factors

## Example Usage

### Python API Client

```python
import requests

# Search for a song
response = requests.get('http://localhost:8000/api/search?query=happy&limit=5')
songs = response.json()

# Get recommendations
rec_response = requests.post('http://localhost:8000/api/recommend/song', json={
    'song_id': songs[0]['id'],
    'n_recommendations': 10,
    'diversity_weight': 0.7
})

recommendations = rec_response.json()
for rec in recommendations['recommendations']:
    print(f"{rec['song']['name']} - {rec['explanation']['explanation']}")
```

## Future Enhancements

- User preference learning
- Collaborative filtering with user data
- Real-time Spotify API integration
- Playlist generation and export
- Advanced visualization of recommendations
- A/B testing of algorithms
- More sophisticated diversity metrics

## Dataset

The system uses Spotify's dataset with:
- 170,655 songs
- 28,682 unique artists
- 2,975 genres
- Data spanning from 1921 to 2020
- 10 audio features per song

## License

This project is for educational purposes as part of Advanced Software for Big Data course.

## Acknowledgments

- Spotify for the dataset
- FastAPI framework
- Scikit-learn library
