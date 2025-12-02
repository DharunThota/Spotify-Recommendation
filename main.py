"""FastAPI backend for Spotify Recommendation System."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from explainability import ExplainabilityEngine
import config


def safe_convert(value, target_type):
    """Safely convert numpy types to native Python types."""
    import numpy as np
    
    if value is None:
        return None
    
    # Convert numpy types to native Python types
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        if target_type == float:
            return float(value)
        elif target_type == int:
            return int(value)
        elif target_type == bool:
            return bool(value)
    
    return target_type(value) if value is not None else None


def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    else:
        return obj


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for song or artist")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")


class SongRecommendationRequest(BaseModel):
    song_id: str = Field(..., description="ID of the song to base recommendations on")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    diversity_weight: float = Field(0.7, ge=0.0, le=1.0, description="Diversity factor")


class MoodRecommendationRequest(BaseModel):
    mood: str = Field(..., description="Target mood: happy, chill, sad, or energetic")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    include_popular: bool = Field(True, description="Bias towards popular songs")


class HybridRecommendationRequest(BaseModel):
    song_ids: List[str] = Field(..., min_items=1, description="List of song IDs")
    mood: Optional[str] = Field(None, description="Optional mood filter")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")


class SongResponse(BaseModel):
    id: str
    name: str
    artists: str
    year: int
    popularity: int
    valence: Optional[float] = None
    energy: Optional[float] = None
    danceability: Optional[float] = None


class RecommendationResponse(BaseModel):
    song: SongResponse
    score: float
    explanation: Optional[dict] = None


class RecommendationListResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    input_songs: Optional[List[SongResponse]] = None
    mood: Optional[str] = None
    total_count: int


# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="A recommendation system with song-based, mood-based, and hybrid recommendations with explainability"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for processors and engines
processor = None
rec_engine = None
explainer = None


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup."""
    global processor, rec_engine, explainer
    
    logger.info("Initializing recommendation system...")
    processor = DataProcessor()
    processor.initialize()
    
    rec_engine = RecommendationEngine(processor)
    explainer = ExplainabilityEngine(processor, rec_engine)
    
    logger.info("Recommendation system ready!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return """
    <html>
        <head><title>Spotify Recommendation System</title></head>
        <body>
            <h1>Spotify Recommendation System API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
        </body>
    </html>
    """


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "total_songs": len(processor.data) if processor else 0,
        "moods_available": list(config.MOOD_CRITERIA.keys())
    }


@app.get("/api/search", response_model=List[SongResponse])
async def search_songs(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """
    Search for songs by name or artist.
    
    - **query**: Search term (song name or artist)
    - **limit**: Maximum number of results to return
    """
    try:
        logger.info(f"Search request: query={query}, limit={limit}")
        results = processor.search_songs(query, limit=limit)
        logger.info(f"Found {len(results)} results")
        
        response = []
        for song in results:
            song_data = processor.get_song_by_id(song['id'])
            response.append(SongResponse(
                id=str(song['id']),
                name=str(song['name']),
                artists=str(song['artists']),
                year=int(song['year']),
                popularity=int(song['popularity']),
                valence=float(song_data['valence']) if song_data is not None else None,
                energy=float(song_data['energy']) if song_data is not None else None,
                danceability=float(song_data['danceability']) if song_data is not None else None
            ))
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend/song", response_model=RecommendationListResponse)
async def recommend_by_song(request: SongRecommendationRequest):
    """
    Get song-based recommendations with explanations.
    
    - **song_id**: Spotify ID of the input song
    - **n_recommendations**: Number of recommendations to return
    - **diversity_weight**: Balance between similarity and diversity (0-1)
    """
    try:
        logger.info(f"Song-based recommendation request: song_id={request.song_id}, n_recs={request.n_recommendations}")
        # Get recommendations
        recommendations = rec_engine.song_based_recommendations(
            request.song_id,
            n_recommendations=request.n_recommendations,
            diversity_weight=request.diversity_weight
        )
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="Song not found or no recommendations available")
        
        # Build response with explanations
        response_list = []
        for i, (idx, score) in enumerate(recommendations):
            logger.debug(f"Processing recommendation {i+1}: idx={idx}, score={score}, type={type(score)}")
            song = processor.get_song_by_index(idx)
            logger.debug(f"Song data types: id={type(song['id'])}, valence={type(song['valence'])}")
            
            # Get explanation
            explanation = explainer.explain_song_recommendation(request.song_id, idx)
            
            # Sanitize explanation for JSON serialization
            explanation = sanitize_for_json(explanation)
            
            song_response = SongResponse(
                id=str(song['id']),
                name=str(song['name']),
                artists=str(song['artists']),
                year=int(song['year']),
                popularity=int(song['popularity']),
                valence=safe_convert(song['valence'], float),
                energy=safe_convert(song['energy'], float),
                danceability=safe_convert(song['danceability'], float)
            )
            
            response_list.append(RecommendationResponse(
                song=song_response,
                score=safe_convert(score, float),
                explanation=explanation
            ))
        
        # Get input song info
        input_song = processor.get_song_by_id(request.song_id)
        input_song_response = SongResponse(
            id=str(request.song_id),
            name=str(input_song['name']),
            artists=str(input_song['artists']),
            year=int(input_song['year']),
            popularity=int(input_song['popularity']),
            valence=float(input_song['valence']),
            energy=float(input_song['energy']),
            danceability=float(input_song['danceability'])
        )
        
        return RecommendationListResponse(
            recommendations=response_list,
            input_songs=[input_song_response],
            total_count=len(response_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend/mood", response_model=RecommendationListResponse)
async def recommend_by_mood(request: MoodRecommendationRequest):
    """
    Get mood-based recommendations with explanations.
    
    - **mood**: Target mood (happy, chill, sad, energetic)
    - **n_recommendations**: Number of recommendations to return
    - **include_popular**: Whether to bias towards popular songs
    """
    if request.mood not in config.MOOD_CRITERIA:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mood. Choose from: {', '.join(config.MOOD_CRITERIA.keys())}"
        )
    
    try:
        logger.info(f"Mood-based recommendation request: mood={request.mood}, n_recs={request.n_recommendations}")
        # Get recommendations
        recommendations = rec_engine.mood_based_recommendations(
            request.mood,
            n_recommendations=request.n_recommendations,
            include_popular=request.include_popular
        )
        logger.info(f"Generated {len(recommendations)} mood recommendations")
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations available for this mood")
        
        # Build response with explanations
        response_list = []
        for i, (idx, score) in enumerate(recommendations):
            logger.info(f"Processing mood recommendation {i+1}: idx={idx}, score={score}, type={type(score)}")
            song = processor.get_song_by_index(idx)
            logger.info(f"Song data: {song['name']} by {song['artists']}")
            
            # Get explanation
            explanation = explainer.explain_mood_recommendation(request.mood, idx)
            
            # Sanitize explanation for JSON serialization
            explanation = sanitize_for_json(explanation)
            
            song_response = SongResponse(
                id=str(song['id']),
                name=str(song['name']),
                artists=str(song['artists']),
                year=int(song['year']),
                popularity=int(song['popularity']),
                valence=safe_convert(song['valence'], float),
                energy=safe_convert(song['energy'], float),
                danceability=safe_convert(song['danceability'], float)
            )
            
            response_list.append(RecommendationResponse(
                song=song_response,
                score=safe_convert(score, float),
                explanation=explanation
            ))
        
        return RecommendationListResponse(
            recommendations=response_list,
            mood=request.mood,
            total_count=len(response_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend/hybrid", response_model=RecommendationListResponse)
async def recommend_hybrid(request: HybridRecommendationRequest):
    """
    Get hybrid recommendations based on multiple songs with optional mood filter.
    
    - **song_ids**: List of Spotify song IDs
    - **mood**: Optional mood filter (happy, chill, sad, energetic)
    - **n_recommendations**: Number of recommendations to return
    """
    if request.mood and request.mood not in config.MOOD_CRITERIA:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mood. Choose from: {', '.join(config.MOOD_CRITERIA.keys())}"
        )
    
    try:
        logger.info(f"Hybrid recommendation request: {len(request.song_ids)} songs, mood={request.mood}")
        # Get recommendations
        recommendations = rec_engine.hybrid_recommendations(
            request.song_ids,
            mood=request.mood,
            n_recommendations=request.n_recommendations
        )
        logger.info(f"Generated {len(recommendations)} hybrid recommendations")
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations available")
        
        # Build response with explanations
        response_list = []
        for idx, score in recommendations:
            song = processor.get_song_by_index(idx)
            
            # Get explanation
            explanation = explainer.explain_hybrid_recommendation(
                request.song_ids, 
                idx,
                request.mood
            )
            
            # Sanitize explanation for JSON serialization
            explanation = sanitize_for_json(explanation)
            
            song_response = SongResponse(
                id=str(song['id']),
                name=str(song['name']),
                artists=str(song['artists']),
                year=int(song['year']),
                popularity=int(song['popularity']),
                valence=safe_convert(song['valence'], float),
                energy=safe_convert(song['energy'], float),
                danceability=safe_convert(song['danceability'], float)
            )
            
            response_list.append(RecommendationResponse(
                song=song_response,
                score=safe_convert(score, float),
                explanation=explanation
            ))
        
        # Get input songs info
        input_songs_response = []
        for song_id in request.song_ids:
            input_song = processor.get_song_by_id(song_id)
            if input_song is not None:
                input_songs_response.append(SongResponse(
                    id=str(song_id),
                    name=str(input_song['name']),
                    artists=str(input_song['artists']),
                    year=int(input_song['year']),
                    popularity=int(input_song['popularity']),
                    valence=float(input_song['valence']),
                    energy=float(input_song['energy']),
                    danceability=float(input_song['danceability'])
                ))
        
        return RecommendationListResponse(
            recommendations=response_list,
            input_songs=input_songs_response,
            mood=request.mood,
            total_count=len(response_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/song/{song_id}", response_model=SongResponse)
async def get_song_details(song_id: str):
    """Get detailed information about a specific song."""
    try:
        song = processor.get_song_by_id(song_id)
        
        if song is None:
            raise HTTPException(status_code=404, detail="Song not found")
        
        return SongResponse(
            id=str(song['id']),
            name=str(song['name']),
            artists=str(song['artists']),
            year=int(song['year']),
            popularity=int(song['popularity']),
            valence=float(song['valence']),
            energy=float(song['energy']),
            danceability=float(song['danceability'])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/moods")
async def get_available_moods():
    """Get list of available moods and their criteria."""
    return {
        "moods": list(config.MOOD_CRITERIA.keys()),
        "criteria": config.MOOD_CRITERIA
    }


# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
