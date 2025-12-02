# Spotify Recommendation System

A comprehensive music recommendation system using big data algorithms, featuring song-based, mood-based, and hybrid recommendations with explainable AI and Spotify Analytics integration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

## Quick Start

```bash
# 1. Install backend dependencies
pip install -r requirements.txt

# 2. Process dataset (first time only)
python data_processor.py

# 3. Start backend server
python main.py

# 4. In a new terminal, start frontend
cd frontend
npm install
npm run dev

# 5. Open browser
# Modern UI: http://localhost:5173
# API Docs: http://localhost:8000/docs
```

## Features

**Song-Based Recommendations**: Find similar songs based on audio features using cosine similarity and K-Means clustering with diversity filtering

**Mood-Based Recommendations**: Discover songs matching 8 different moods with weighted feature scoring:
- Happy, Chill, Sad, Energetic, Romantic, Focus, Party, Melancholic

**Hybrid Recommendations**: Combine multiple songs with optional mood filtering for personalized playlists

**Spotify Analytics Integration**: Connect your Spotify account to view:
- Top tracks and artists analysis
- Listening history insights
- Personalized Spotify Wrapped-style visualizations
- Audio feature distributions

**Explainable AI**: Every recommendation comes with human-readable explanations showing why songs were suggested

**Popular Songs Filter**: Toggle to show only popular tracks (popularity >= 40)

**Modern React UI**: Clean, responsive interface with:
- Beautiful gradient designs
- Smooth animations and transitions
- Mood-based color coding
- Interactive song cards with detailed modals
- Real-time search with debouncing

**Dual Processing Engines**: Toggle between **Pandas** (fast for small datasets) and **PySpark** (scalable for big data)

**Advanced Big Data Algorithms**:
- K-Means Clustering (75 clusters) for efficient candidate selection
- Cosine Similarity for content-based filtering
- Weighted feature scoring with Gaussian-like distribution
- Cluster-based diversity filtering for sonic variety
- Artist diversity penalties to avoid repetition
- Feature engineering and normalization
- Distributed computing support via PySpark

## Project Structure

```
ASBD/
├── data/                          # Dataset files
│   ├── data.csv                   # Main song dataset (170K+ songs)
│   ├── data_by_artist.csv         # Artist aggregated data
│   ├── data_by_genres.csv         # Genre aggregated data
│   ├── data_by_year.csv           # Year aggregated data
│   └── data_w_genres.csv          # Songs with genre information
├── frontend/                      # React Frontend (NEW!)
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── services/              # API integration
│   │   ├── App.jsx                # Main app
│   │   └── main.jsx               # Entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── static/                        # Legacy vanilla JS frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── config.py                      # Configuration settings
├── data_processor.py              # Data loading and preprocessing
├── recommendation_engine.py       # Recommendation algorithms
├── explainability.py              # Explanation generation
├── main.py                        # FastAPI backend
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Backend Setup

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify data files are in the `data/` folder**

3. **(Optional) Configure Processing Engine**

Choose between Pandas and PySpark in `config.py`:

```python
USE_PYSPARK = False  # Set to True for PySpark, False for Pandas
```

For PySpark installation:
```bash
pip install pyspark
```

📖 **See [PYSPARK_GUIDE.md](PYSPARK_GUIDE.md) for detailed PySpark configuration**

### Frontend Setup (React)

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install Node.js dependencies**:
```bash
npm install
```

## Usage

### 1. Initialize the System (First Run)

Process and prepare the dataset:

```bash
python data_processor.py
```

This will:
- Load all CSV files (using Pandas or PySpark based on config)
- Normalize audio features
- Extract mood classifications
- Perform K-Means clustering
- Create indexed lookup structures
- Save processed data to `processed_data.pkl` (Pandas) or `processed_data_pyspark.pkl` (PySpark)

### 2. (Optional) Benchmark Performance

Compare Pandas vs PySpark performance:

```bash
python benchmark.py
```

### 3. Start the FastAPI Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the React Frontend (in a new terminal)

```bash
cd frontend
npm run dev
```

### 4. Access the Application

**Option 1: React Frontend (Recommended)**
- **Modern UI**: http://localhost:3000
- Clean, responsive React interface with better UX

**Option 2: Legacy Vanilla JS Frontend**
- **Basic UI**: http://localhost:8000
- Simple HTML/CSS/JS interface

**API Documentation**
- **Swagger UI**: http://localhost:8000/docs
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
  "mood": "happy|chill|sad|energetic|romantic|focus|party|melancholic",
  "n_recommendations": 10,
  "include_popular": true
}
```

### Hybrid Recommendations
```
POST /api/recommend/hybrid
Body: {
  "song_ids": ["id1", "id2", ...],
  "mood": "happy|chill|sad|energetic|romantic|focus|party|melancholic" (optional),
  "n_recommendations": 10
}
```

### Spotify Analytics (OAuth Required)
```
GET /api/auth/login          # Initiate Spotify OAuth
GET /api/auth/callback       # OAuth callback handler
GET /api/auth/status         # Check authentication status
GET /api/analytics/wrapped-insights?time_range={short_term|medium_term|long_term}
```

## How It Works

### 1. Data Processing
- Loads 170K+ songs with audio features (valence, energy, danceability, etc.)
- Normalizes features using StandardScaler
- Creates indexed lookup structures for fast retrieval

### 2. Mood Classification
Songs are classified into 8 moods using weighted feature scoring with 6 audio features each:

- **Happy**: High valence + energy + danceability, upbeat tempo (100-180 BPM)
- **Chill**: Low energy, moderate tempo (60-120 BPM), high acousticness, quieter
- **Sad**: Low valence + energy, slower tempo (60-100 BPM), acoustic sound
- **Energetic**: High energy + tempo (120-200 BPM) + danceability, loud, positive
- **Romantic**: Moderate valence + energy, acoustic (30-90%), slower (70-120 BPM)
- **Focus**: Moderate energy, highly instrumental (50-100%), minimal vocals, steady tempo
- **Party**: High energy + danceability + valence, fast (120-180 BPM), loud
- **Melancholic**: Low valence (10-40%) + energy, acoustic, slow tempo (60-100 BPM)

Each mood uses weighted features where more important characteristics have higher weights (e.g., valence: 2.0, energy: 1.5) for better matching accuracy.

### 3. Clustering
- K-Means clustering groups similar songs
- 75 clusters for diverse groupings
- Used for efficient candidate selection

### 4. Similarity Computation
- Cosine similarity between feature vectors
- Considers 10 audio features
- Applies diversity penalties to avoid repetition

### 5. Weighted Mood Scoring
- Uses Gaussian-like scoring for in-range values (closer to center = higher score)
- Applies exponential decay penalty for out-of-range values
- Different feature weights per mood (e.g., valence weighted 2.0x for happy mood)
- Normalized scoring across all features

### 6. Cluster-Based Diversity
- Round-robin selection from different clusters
- Ensures sonic variety within same mood
- Prevents recommendation monotony
- Maintains audio feature diversity

### 7. Explainability
Generates explanations by:
- Analyzing feature similarities between songs
- Finding common attributes (artists, era, mode, key)
- Calculating weighted mood fit scores
- Identifying shared characteristics (same decade, genre, tempo range)
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

**Backend:**
- **FastAPI**: Modern async web framework for APIs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms (K-Means, cosine similarity)
- **Uvicorn**: ASGI server for FastAPI
- **Spotipy**: Spotify Web API wrapper for Python
- **PySpark** (optional): Distributed data processing
- **Python-dotenv**: Environment variable management

**Frontend:**
- **React 18**: Modern UI library with hooks
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API calls
- **Lucide React**: Beautiful icon library
- **CSS3**: Modern styling with gradients, animations, flexbox/grid

## Spotify Analytics Setup

To use the Spotify Analytics feature:

1. **Create a Spotify App**:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Get your Client ID and Client Secret

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/api/auth/callback
   ```

3. **Configure Redirect URI**:
   - In Spotify Developer Dashboard, add redirect URI:
   - `http://127.0.0.1:8000/api/auth/callback`

4. **Clear cache if needed**:
   ```bash
   rm .spotify_cache
   ```

## Configuration

Edit `config.py` to customize:

### Recommendation Settings
- `N_RECOMMENDATIONS`: Default number of recommendations (10)
- `MIN_SIMILARITY_THRESHOLD`: Minimum similarity score (0.3)
- `DIVERSITY_FACTOR`: Balance between similarity and diversity (0.7)

### Clustering Parameters
- `N_CLUSTERS`: Number of K-Means clusters (75)
- `RANDOM_STATE`: Random seed for reproducibility (42)

### Mood Criteria
- `MOOD_CRITERIA`: Feature ranges for each of 8 moods (6 features per mood)
- `MOOD_FEATURE_WEIGHTS`: Importance weights for each feature per mood

### Processing Engine
- `USE_PYSPARK`: Toggle between Pandas (False) and PySpark (True)
- `PYSPARK_CONFIG`: Spark configuration (memory, cores, etc.)

### Spotify API
- `SPOTIFY_CLIENT_ID`: Your Spotify app client ID
- `SPOTIFY_CLIENT_SECRET`: Your Spotify app client secret
- `SPOTIFY_REDIRECT_URI`: OAuth callback URL
- `SPOTIFY_SCOPES`: Required API permissions

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

## Key Features in Detail

### 1. Advanced Mood System
- **8 Comprehensive Moods**: Happy, Chill, Sad, Energetic, Romantic, Focus, Party, Melancholic
- **6 Audio Features per Mood**: Each mood considers 6 carefully selected audio features
- **Weighted Scoring**: Important features weighted higher (e.g., valence 2.0x for happy)
- **Smart Thresholds**: Gaussian-like scoring rewards center values, penalties for outliers

### 2. Spotify Analytics Dashboard
- **OAuth Integration**: Secure authentication with Spotify
- **Top Tracks Analysis**: Your most played songs with audio feature breakdowns
- **Top Artists**: Favorite artists with genre insights
- **Listening Patterns**: Time-range analysis (short/medium/long term)
- **Audio Profiles**: Visual representations of your music taste
- **Wrapped-Style Insights**: Personalized music statistics

### 3. Intelligent Diversity Filtering
- **Cluster-Based Selection**: Round-robin from different sonic clusters
- **Artist Diversity**: Penalties for same-artist repetition (50% penalty)
- **Popularity Boost**: Slight boost for popular tracks (1 + popularity/1000)
- **Multi-Stage Filtering**: Mood fit → cluster diversity → artist diversity

### 4. Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Search**: Debounced search with instant results
- **Song Modals**: Detailed information with audio feature visualizations
- **Load More**: Pagination for better performance
- **Empty States**: Helpful messages when no results found
- **Error Handling**: User-friendly error messages with suggestions

## Future Enhancements

- Sequential pattern mining for listening history
- Collaborative filtering with user similarity
- Playlist generation and Spotify export
- Advanced visualization dashboards
- A/B testing framework for algorithms
- User feedback loop for preference learning
- Genre-based recommendations
- Tempo/BPM based workout playlists
- Multi-language support

## Dataset

The system uses Spotify's dataset with:
- 170,655 songs
- 28,682 unique artists
- 2,975 genres
- Data spanning from 1921 to 2020
- 10 audio features per song

## License

This project is for educational purposes as part of Analytics and Systems of Big Data course.

## Troubleshooting

### "Popular Only" toggle shows no results
- The threshold is set to popularity >= 40
- Some moods may have fewer popular songs
- Try disabling the filter or selecting different moods

### Hybrid recommendations with mood filter fails
- Ensure the backend server is restarted after updates
- Check that `processed_data.pkl` contains all 8 moods
- Delete cache and restart: `rm processed_data.pkl && python main.py`

### Spotify Analytics not working
- Verify `.env` file has correct credentials
- Check redirect URI in Spotify Developer Dashboard
- Clear cache: `rm .spotify_cache`
- Ensure your Spotify account has activity to analyze

### Frontend not connecting to backend
- Verify backend is running on port 8000
- Check CORS settings in `main.py`
- Ensure API_URL in frontend matches backend URL

## Tech Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI | RESTful API with async support |
| **Data Processing** | Pandas / PySpark | Dataset manipulation and analysis |
| **Machine Learning** | Scikit-learn | K-Means clustering, similarity |
| **Spotify Integration** | Spotipy | OAuth and Web API access |
| **Frontend Framework** | React 18 + Vite | Modern UI with fast dev server |
| **Styling** | CSS3 | Gradients, animations, responsive |
| **Icons** | Lucide React | Beautiful, consistent icons |
| **HTTP Client** | Axios | API communication |
| **Server** | Uvicorn | ASGI server for FastAPI |

## Performance Metrics

- **Dataset Size**: 170,655 songs, 28,682 artists, 2,975 genres
- **Processing Time**: ~30-60 seconds (Pandas), ~20-40 seconds (PySpark)
- **Recommendation Speed**: <100ms for song-based, <200ms for mood-based
- **Memory Usage**: ~500MB (Pandas), ~1-2GB (PySpark with 4GB driver)
- **Clustering**: 75 clusters for optimal diversity vs performance
- **API Response Time**: <500ms average (including explanations)

## Acknowledgments

- **Spotify** for the comprehensive music dataset
- **FastAPI** framework for elegant API design
- **Scikit-learn** for machine learning algorithms
