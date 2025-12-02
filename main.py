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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from data_processor import create_data_processor
from recommendation_engine import RecommendationEngine
from explainability import ExplainabilityEngine
from spotify_analytics import SpotifyAnalytics
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
    song_ids: List[str] = Field(..., min_length=1, description="List of song IDs")
    mood: Optional[str] = Field(None, description="Optional mood filter")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")


class SequenceRecommendationRequest(BaseModel):
    song_id: str = Field(..., description="Current song ID")
    recent_context: List[str] = Field([], description="Recently played song IDs for sequence mining")
    sequence_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for sequence patterns vs audio features")
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
analytics_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup."""
    global processor, rec_engine, explainer, analytics_engine
    
    logger.info("Initializing recommendation system...")
    logger.info(f"Using {'PySpark' if config.USE_PYSPARK else 'Pandas'} for data processing")
    
    processor = create_data_processor()
    processor.initialize()
    
    rec_engine = RecommendationEngine(processor)
    explainer = ExplainabilityEngine(processor, rec_engine)
    analytics_engine = SpotifyAnalytics()
    
    logger.info("Recommendation system ready!")
    logger.info(f"RecommendationEngine methods: {[m for m in dir(rec_engine) if not m.startswith('_')]}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global processor
    
    logger.info("Shutting down recommendation system...")
    if config.USE_PYSPARK and hasattr(processor, 'stop_spark'):
        processor.stop_spark()
    logger.info("Cleanup complete!")


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
        "processing_engine": "PySpark" if config.USE_PYSPARK else "Pandas",
        "total_songs": len(processor.data if hasattr(processor, 'data') else processor.data_pandas) if processor else 0,
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
        logger.error(f"Error in mood recommendation: {str(e)}", exc_info=True)
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
        logger.info(f"Generated {len(recommendations)} hybrid recommendations with mood={request.mood}")
        
        if not recommendations:
            logger.warning(f"No hybrid recommendations found for songs={request.song_ids}, mood={request.mood}")
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found. Try removing the mood filter or selecting different songs."
            )
        
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
        logger.error(f"Error in hybrid recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend/sequence-aware", response_model=RecommendationListResponse)
async def recommend_sequence_aware(request: SequenceRecommendationRequest):
    """
    Get sequence-aware recommendations combining content-based and collaborative filtering.
    Uses listening patterns from recent_context to predict next songs.
    
    - **song_id**: Current/seed song ID
    - **recent_context**: List of recently played song IDs for sequence mining
    - **sequence_weight**: Balance between sequence patterns (collaborative) and audio features (content-based)
    - **n_recommendations**: Number of recommendations to return
    """
    try:
        logger.info(f"Sequence-aware recommendation: song_id={request.song_id}, context={len(request.recent_context)} songs, weight={request.sequence_weight}")
        
        # Convert song_id to index
        input_song_idx = processor.song_id_to_idx.get(request.song_id)
        if input_song_idx is None:
            raise HTTPException(status_code=404, detail="Song not found in dataset")
        
        # Build context indices from recent_context
        context_indices = []
        
        # Add recent context songs first (in order)
        for song_id in request.recent_context:
            idx = processor.song_id_to_idx.get(song_id)
            if idx is not None and idx != input_song_idx:
                context_indices.append(idx)
        
        # Add current song at the end
        context_indices.append(input_song_idx)
        
        logger.info(f"Context indices: {len(context_indices)} songs (current song + {len(request.recent_context)} context)")
        
        # Create a temporary sequence miner if we have enough context
        sequence_miner = None
        if len(context_indices) >= 3:  # Need at least 3 songs to mine meaningful patterns
            try:
                from sequence_mining import SequentialPatternMiner
                from collections import Counter
                import pandas as pd
                
                sequence_miner = SequentialPatternMiner(
                    min_support=0.1,  # Very low threshold for small context
                    max_gap=3,
                    session_gap_minutes=30
                )
                
                # Convert indices to song IDs
                context_song_ids = [processor.get_song_by_index(idx)['id'] for idx in context_indices]
                
                # Instead of mining from single session, create multiple synthetic sessions
                # by treating different subsequences as separate sessions
                # This allows PrefixSpan to find patterns with proper support
                session_records = []
                session_counter = 0
                
                # Create overlapping sessions from the context
                for i in range(len(context_song_ids)):
                    for j in range(i + 2, min(i + 5, len(context_song_ids) + 1)):
                        subsequence = context_song_ids[i:j]
                        if len(subsequence) >= 2:
                            for k, song_id in enumerate(subsequence):
                                session_records.append({
                                    'session_id': f'session_{session_counter}',
                                    'song_id': song_id,
                                    'played_at': pd.Timestamp.now() + pd.Timedelta(minutes=k)
                                })
                            session_counter += 1
                
                # Also add the full sequence as a session
                for k, song_id in enumerate(context_song_ids):
                    session_records.append({
                        'session_id': 'full_context',
                        'song_id': song_id,
                        'played_at': pd.Timestamp.now() + pd.Timedelta(minutes=k)
                    })
                
                if len(session_records) >= 4:  # Need at least 2 sessions with 2 songs each
                    # fit() expects list of dicts with 'id' and 'played_at'
                    # But our records have 'song_id', so we need to rename or adjust
                    listening_history = [{'id': rec['song_id'], 'played_at': rec['played_at']} 
                                        for rec in session_records]
                    sequence_miner.fit(listening_history)
                    logger.info(f"Created {session_counter + 1} synthetic sessions, mined {len(sequence_miner.patterns)} patterns")
                else:
                    logger.info(f"Not enough context ({len(context_song_ids)} songs) to mine patterns")
                    sequence_miner = None
            except Exception as e:
                logger.warning(f"Could not mine patterns from context: {e}", exc_info=True)
                sequence_miner = None
        
        # Get sequence-aware recommendations
        if sequence_miner and sequence_miner.patterns:
            logger.info(f"Using sequence-aware recommendations with weight={request.sequence_weight}")
            recommendations = rec_engine.sequence_aware_recommendations(
                context_indices,
                sequence_miner=sequence_miner,
                num_recommendations=request.n_recommendations,
                sequence_weight=request.sequence_weight
            )
        # else:
        #     # Fallback to hybrid recommendations if no patterns
        #     logger.info("Using hybrid recommendations (no patterns available)")
        #     # Convert indices to song IDs for hybrid method
        #     context_song_ids = [processor.get_song_by_index(idx)['id'] for idx in context_indices]
        #     recommendations = rec_engine.hybrid_recommendations(
        #         context_song_ids,
        #         n_recommendations=request.n_recommendations
        #     )
        
        logger.info(f"Received {len(recommendations) if recommendations else 0} recommendations from engine")
        
        if not recommendations:
            logger.warning(f"No recommendations generated for context")
            raise HTTPException(status_code=404, detail="No recommendations available")
        
        logger.info(f"Generated {len(recommendations)} sequence-aware recommendations")
        
        # Build response with explanations
        response_list = []
        
        for idx, score in recommendations:
            song = processor.get_song_by_index(idx)
            
            # Get explanation
            if sequence_miner and sequence_miner.patterns:
                try:
                    explanation = explainer.explain_sequence_recommendation(
                        recommended_idx=idx,
                        context_song_indices=context_indices,
                        sequence_miner=sequence_miner,
                        content_score=score * (1 - request.sequence_weight),  # Approximate content portion
                        sequence_score=score * request.sequence_weight  # Approximate sequence portion
                    )
                except Exception as e:
                    logger.warning(f"Could not generate sequence explanation: {e}")
                    explanation = explainer.explain_song_recommendation(request.song_id, idx)
            else:
                # Fallback to regular song-based explanation
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
        
        # Get input songs info (all context songs)
        input_songs_response = []
        for idx in context_indices[:3]:  # Show up to 3 context songs
            song = processor.get_song_by_index(idx)
            if song is not None:
                input_songs_response.append(SongResponse(
                    id=str(song['id']),
                    name=str(song['name']),
                    artists=str(song['artists']),
                    year=int(song['year']),
                    popularity=int(song['popularity']),
                    valence=safe_convert(song['valence'], float),
                    energy=safe_convert(song['energy'], float),
                    danceability=safe_convert(song['danceability'], float)
                ))
        
        return RecommendationListResponse(
            recommendations=response_list,
            input_songs=input_songs_response,
            mood=None,
            total_count=len(response_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sequence-aware recommendations: {e}", exc_info=True)
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


# ================== SPOTIFY ANALYTICS ENDPOINTS ==================

@app.get("/api/analytics/check-auth")
async def check_auth_status():
    """Check if user is authenticated (has valid cached token)."""
    try:
        auth_manager = analytics_engine.get_auth_manager()
        token_info = auth_manager.get_cached_token()
        
        if token_info and not auth_manager.is_token_expired(token_info):
            # Re-authenticate with cached token if not already authenticated
            if not analytics_engine.sp:
                result = analytics_engine.authenticate(token_info)
                if result['success']:
                    return {
                        "authenticated": True,
                        "user_id": result['user_id'],
                        "display_name": result['display_name']
                    }
            else:
                # Already authenticated
                user_profile = analytics_engine.sp.current_user()
                return {
                    "authenticated": True,
                    "user_id": analytics_engine.user_id,
                    "display_name": user_profile.get('display_name', 'User')
                }
        
        return {"authenticated": False}
    except Exception as e:
        logger.error(f"Error checking auth status: {str(e)}")
        return {"authenticated": False}


@app.get("/api/analytics/auth-url")
async def get_spotify_auth_url():
    """Get Spotify OAuth authorization URL."""
    try:
        auth_manager = analytics_engine.get_auth_manager()
        auth_url = auth_manager.get_authorize_url()
        return {"auth_url": auth_url}
    except Exception as e:
        logger.error(f"Error generating auth URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/callback")
async def spotify_callback(code: str):
    """Handle Spotify OAuth callback."""
    try:
        auth_manager = analytics_engine.get_auth_manager()
        token_info = auth_manager.get_access_token(code)
        
        # Authenticate with the token
        result = analytics_engine.authenticate(token_info)
        
        if result['success']:
            # Redirect to analytics page with success
            return HTMLResponse(content=f"""
                <html>
                    <head>
                        <title>Spotify Authentication</title>
                        <script>
                            window.opener.postMessage({{
                                type: 'spotify_auth_success',
                                user_id: '{result['user_id']}',
                                display_name: '{result['display_name']}'
                            }}, '*');
                            window.close();
                        </script>
                    </head>
                    <body>
                        <h1>Authentication Successful!</h1>
                        <p>You can close this window.</p>
                    </body>
                </html>
            """)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Authentication failed'))
            
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(
    time_range: str = Query('medium_term', pattern='^(short_term|medium_term|long_term)$')
):
    """Get comprehensive analytics dashboard for authenticated user."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Please authenticate with Spotify first."
            )
        
        dashboard = analytics_engine.get_comprehensive_dashboard(time_range)
        return sanitize_for_json(dashboard)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/top-tracks")
async def get_top_tracks(
    time_range: str = Query('medium_term', pattern='^(short_term|medium_term|long_term)$'),
    limit: int = Query(20, ge=1, le=50)
):
    """Get user's top tracks."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        tracks = analytics_engine.get_top_tracks(time_range, limit)
        return sanitize_for_json(tracks)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching top tracks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/top-artists")
async def get_top_artists(
    time_range: str = Query('medium_term', pattern='^(short_term|medium_term|long_term)$'),
    limit: int = Query(20, ge=1, le=50)
):
    """Get user's top artists."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        artists = analytics_engine.get_top_artists(time_range, limit)
        return sanitize_for_json(artists)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching top artists: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/listening-patterns")
async def get_listening_patterns(limit: int = Query(50, ge=10, le=100)):
    """Analyze user's listening patterns."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        tracks = analytics_engine.get_recently_played(limit)
        analysis = analytics_engine.analyze_listening_patterns(tracks)
        
        return sanitize_for_json(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/associations")
async def get_association_mining(
    limit: int = Query(50, ge=10, le=100),
    min_support: float = Query(0.15, ge=0.01, le=0.5),
    min_confidence: float = Query(0.6, ge=0.1, le=1.0)
):
    """Perform association rule mining on user's listening history."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        tracks = analytics_engine.get_recently_played(limit)
        associations = analytics_engine.perform_association_mining(
            tracks,
            min_support,
            min_confidence
        )
        
        return sanitize_for_json(associations)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in association mining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/visualizations")
async def get_visualizations(
    time_range: str = Query('medium_term', pattern='^(short_term|medium_term|long_term)$')
):
    """Get visualization data for analytics dashboard."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        tracks = analytics_engine.get_top_tracks(time_range, 50)
        analysis = analytics_engine.analyze_listening_patterns(tracks)
        visualizations = analytics_engine.generate_visualizations(analysis)
        
        return visualizations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/wrapped-insights")
async def get_wrapped_insights(
    time_range: str = Query('medium_term', pattern='^(short_term|medium_term|long_term)$')
):
    """Get Spotify-Wrapped style comprehensive insights."""
    try:
        if not analytics_engine.sp:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        insights = analytics_engine.get_wrapped_insights(time_range)
        return sanitize_for_json(insights)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating wrapped insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
