"""Spotify Analytics Module with OAuth and Listening History Analysis."""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.graph_objects as go
import plotly.express as px
import config

logger = logging.getLogger(__name__)


class SpotifyAnalytics:
    """Analytics engine for Spotify user listening history."""
    
    def __init__(self, cache_path=".spotify_cache"):
        """Initialize Spotify Analytics with OAuth."""
        self.cache_path = cache_path
        self.sp = None
        self.user_id = None
        
    def get_auth_manager(self):
        """Get Spotify OAuth manager."""
        return SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=" ".join(config.SPOTIFY_SCOPES),
            cache_path=self.cache_path
        )
    
    def authenticate(self, token_info: Optional[Dict] = None):
        """Authenticate user with Spotify."""
        try:
            if token_info:
                self.sp = spotipy.Spotify(auth=token_info['access_token'])
            else:
                auth_manager = self.get_auth_manager()
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Get user profile
            user_profile = self.sp.current_user()
            self.user_id = user_profile['id']
            
            logger.info(f"Authenticated user: {self.user_id}")
            return {
                "success": True,
                "user_id": self.user_id,
                "display_name": user_profile.get('display_name', 'User')
            }
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """Get user's recently played tracks."""
        if not self.sp:
            raise Exception("Not authenticated. Please authenticate first.")
        
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            tracks = []
            
            for item in results['items']:
                track = item['track']
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_ids': [artist['id'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'played_at': item['played_at'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                })
            
            return tracks
        except Exception as e:
            logger.error(f"Error fetching recently played: {str(e)}")
            raise
    
    def get_top_tracks(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict]:
        """Get user's top tracks.
        
        time_range: short_term (4 weeks), medium_term (6 months), long_term (years)
        """
        if not self.sp:
            raise Exception("Not authenticated. Please authenticate first.")
        
        try:
            results = self.sp.current_user_top_tracks(
                time_range=time_range,
                limit=limit
            )
            
            tracks = []
            for track in results['items']:
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_ids': [artist['id'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                })
            
            return tracks
        except Exception as e:
            logger.error(f"Error fetching top tracks: {str(e)}")
            raise
    
    def get_top_artists(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict]:
        """Get user's top artists."""
        if not self.sp:
            raise Exception("Not authenticated. Please authenticate first.")
        
        try:
            results = self.sp.current_user_top_artists(
                time_range=time_range,
                limit=limit
            )
            
            artists = []
            for artist in results['items']:
                artists.append({
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres'],
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total']
                })
            
            return artists
        except Exception as e:
            logger.error(f"Error fetching top artists: {str(e)}")
            raise
    
    def get_audio_features_batch(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for multiple tracks."""
        if not self.sp:
            raise Exception("Not authenticated. Please authenticate first.")
        
        try:
            # Spotify API allows max 100 tracks per request
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                try:
                    features = self.sp.audio_features(batch)
                    all_features.extend([f for f in features if f is not None])
                except Exception as api_error:
                    # Audio features endpoint may not be available
                    logger.warning(f"Audio features unavailable: {str(api_error)}")
                    # Return empty features for unavailable tracks
                    for track_id in batch:
                        all_features.append({
                            'id': track_id,
                            'valence': None,
                            'energy': None,
                            'danceability': None,
                            'acousticness': None,
                            'instrumentalness': None,
                            'liveness': None,
                            'speechiness': None,
                            'tempo': None,
                            'loudness': None
                        })
            
            return all_features
        except Exception as e:
            logger.error(f"Error fetching audio features: {str(e)}")
            # Return empty features dict to allow analysis to continue
            return [{
                'id': tid,
                'valence': None,
                'energy': None,
                'danceability': None,
                'acousticness': None,
                'instrumentalness': None,
                'liveness': None,
                'speechiness': None,
                'tempo': None,
                'loudness': None
            } for tid in track_ids]
    
    def analyze_listening_patterns(self, tracks: List[Dict]) -> Dict:
        """Analyze listening patterns from track data."""
        if not tracks:
            return {}
        
        df = pd.DataFrame(tracks)
        
        # Get audio features
        track_ids = df['id'].tolist()
        audio_features = self.get_audio_features_batch(track_ids)
        features_df = pd.DataFrame(audio_features)
        
        # Merge with track data
        df = df.merge(features_df, left_on='id', right_on='id', how='left')
        
        # Check if audio features are available
        has_audio_features = 'valence' in df.columns and df['valence'].notna().any()
        
        analysis = {
            'total_tracks': len(df),
            'unique_tracks': df['id'].nunique(),
            'unique_artists': df['artist'].nunique(),
            'avg_popularity': float(df['popularity'].mean()) if 'popularity' in df.columns else 0,
            'avg_duration_min': float(df['duration_ms'].mean() / 60000) if 'duration_ms' in df.columns else 0,
            'has_audio_features': has_audio_features,
            
            # Audio feature averages (only if available)
            'audio_features': {
                'valence': float(df['valence'].mean()) if has_audio_features else None,
                'energy': float(df['energy'].mean()) if has_audio_features else None,
                'danceability': float(df['danceability'].mean()) if has_audio_features else None,
                'acousticness': float(df['acousticness'].mean()) if has_audio_features else None,
                'instrumentalness': float(df['instrumentalness'].mean()) if has_audio_features else None,
                'liveness': float(df['liveness'].mean()) if has_audio_features else None,
                'speechiness': float(df['speechiness'].mean()) if has_audio_features else None,
                'tempo': float(df['tempo'].mean()) if has_audio_features else None,
                'loudness': float(df['loudness'].mean()) if has_audio_features else None
            },
            
            # Top items
            'top_artists': df['artist'].value_counts().head(10).to_dict(),
            'top_tracks': [
                {'name': name, 'artist': artist, 'count': int(count)}
                for (name, artist), count in df.groupby(['name', 'artist']).size().sort_values(
                    ascending=False
                ).head(10).items()
            ]
        }
        
        return analysis
    
    def detect_music_moods(self, audio_features: Dict) -> List[str]:
        """Detect dominant moods based on audio features."""
        moods = []
        
        # Safely get audio features with defaults
        valence = audio_features.get('valence')
        energy = audio_features.get('energy')
        tempo = audio_features.get('tempo')
        
        # Only detect moods if we have the necessary features
        if valence is not None and energy is not None:
            if valence > 0.6 and energy > 0.5:
                moods.append('happy')
            if energy < 0.5 and tempo is not None and tempo < 100:
                moods.append('chill')
            if valence < 0.4 and energy < 0.5:
                moods.append('sad')
            if energy > 0.7 and tempo is not None and tempo > 120:
                moods.append('energetic')
        
        return moods if moods else ['neutral']
    
    def perform_association_mining(
        self,
        tracks: List[Dict],
        min_support: float = 0.1,
        min_confidence: float = 0.5
    ) -> Dict:
        """Perform association rule mining on listening patterns.
        
        Discovers patterns like:
        - If user listens to Artist A, they likely listen to Artist B
        - If user likes Genre X, they likely enjoy Genre Y
        - Song features that commonly appear together
        """
        if not tracks or len(tracks) < 10:
            return {"error": "Not enough data for association mining"}
        
        try:
            df = pd.DataFrame(tracks)
            
            # Get audio features
            track_ids = df['id'].tolist()
            audio_features = self.get_audio_features_batch(track_ids)
            features_df = pd.DataFrame(audio_features)
            df = df.merge(features_df, left_on='id', right_on='id', how='left')
            
            # Create transactions for different analyses
            
            # 1. Artist co-listening patterns
            artist_transactions = []
            # Group by listening session (e.g., by day)
            if 'played_at' in df.columns:
                df['date'] = pd.to_datetime(df['played_at']).dt.date
                for date, group in df.groupby('date'):
                    artists = group['artist'].unique().tolist()
                    if len(artists) > 1:
                        artist_transactions.append(artists)
            
            artist_rules = None
            if len(artist_transactions) > 5:
                artist_rules = self._mine_associations(
                    artist_transactions,
                    min_support,
                    min_confidence,
                    "Artists"
                )
            
            # 2. Mood-based patterns
            mood_transactions = []
            for _, row in df.iterrows():
                # Only detect moods if audio features are available
                if row.get('valence') is not None and row.get('energy') is not None:
                    moods = self.detect_music_moods({
                        'valence': row.get('valence', 0.5),
                        'energy': row.get('energy', 0.5),
                        'tempo': row.get('tempo', 120)
                    })
                    if moods:
                        mood_transactions.append(moods)
            
            mood_patterns = None
            if len(mood_transactions) > 5:
                mood_patterns = self._mine_associations(
                    mood_transactions,
                    min_support,
                    min_confidence,
                    "Moods"
                )
            
            # 3. Audio feature patterns (discretized)
            feature_transactions = []
            for _, row in df.iterrows():
                features = []
                
                # Only process audio features if they're available
                valence = row.get('valence')
                energy = row.get('energy')
                danceability = row.get('danceability')
                acousticness = row.get('acousticness')
                
                if valence is not None:
                    if valence > 0.6:
                        features.append('high_valence')
                    elif valence < 0.4:
                        features.append('low_valence')
                    
                if energy is not None:
                    if energy > 0.6:
                        features.append('high_energy')
                    elif energy < 0.4:
                        features.append('low_energy')
                    
                if danceability is not None and danceability > 0.6:
                    features.append('danceable')
                    
                if acousticness is not None and acousticness > 0.5:
                    features.append('acoustic')
                    
                if len(features) > 1:
                    feature_transactions.append(features)
            
            feature_patterns = None
            if len(feature_transactions) > 5:
                feature_patterns = self._mine_associations(
                    feature_transactions,
                    min_support,
                    min_confidence,
                    "Audio Features"
                )
            
            return {
                "artist_associations": artist_rules,
                "mood_patterns": mood_patterns,
                "feature_patterns": feature_patterns,
                "summary": {
                    "total_transactions_artists": len(artist_transactions),
                    "total_transactions_moods": len(mood_transactions),
                    "total_transactions_features": len(feature_transactions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in association mining: {str(e)}")
            return {"error": str(e)}
    
    def _mine_associations(
        self,
        transactions: List[List[str]],
        min_support: float,
        min_confidence: float,
        label: str
    ) -> Optional[Dict]:
        """Mine association rules from transactions."""
        try:
            # Encode transactions
            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_array, columns=te.columns_)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(
                df,
                min_support=min_support,
                use_colnames=True
            )
            
            if frequent_itemsets.empty:
                return None
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )
            
            if rules.empty:
                return None
            
            # Format rules for output
            top_rules = rules.nlargest(10, 'lift')
            
            formatted_rules = []
            for _, rule in top_rules.iterrows():
                # Convert frozensets to lists for JSON serialization
                antecedents_list = sorted(list(rule['antecedents']))
                consequents_list = sorted(list(rule['consequents']))
                
                formatted_rules.append({
                    'antecedents': antecedents_list,
                    'consequents': consequents_list,
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'description': f"If you listen to {', '.join(antecedents_list)}, "
                                 f"you're likely to enjoy {', '.join(consequents_list)} "
                                 f"(confidence: {rule['confidence']:.1%})"
                })
            
            return {
                'rules': formatted_rules,
                'total_rules': len(rules),
                'avg_confidence': float(rules['confidence'].mean()),
                'avg_lift': float(rules['lift'].mean())
            }
            
        except Exception as e:
            logger.error(f"Error mining {label} associations: {str(e)}")
            return None
    
    def generate_visualizations(self, analysis: Dict) -> Dict:
        """Generate Plotly visualizations for the analytics dashboard."""
        try:
            visualizations = {}
            
            # 1. Audio Features Radar Chart
            if 'audio_features' in analysis:
                features = analysis['audio_features']
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[
                        features['valence'],
                        features['energy'],
                        features['danceability'],
                        features['acousticness'],
                        features['speechiness']
                    ],
                    theta=['Valence', 'Energy', 'Danceability', 'Acousticness', 'Speechiness'],
                    fill='toself',
                    line_color='#1DB954'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Your Music Personality"
                )
                visualizations['audio_features_radar'] = fig_radar.to_json()
            
            # 2. Top Artists Bar Chart
            if 'top_artists' in analysis:
                top_artists = analysis['top_artists']
                fig_artists = go.Figure(data=[
                    go.Bar(
                        x=list(top_artists.values()),
                        y=list(top_artists.keys()),
                        orientation='h',
                        marker_color='#1DB954'
                    )
                ])
                fig_artists.update_layout(
                    title="Your Top Artists",
                    xaxis_title="Play Count",
                    yaxis_title="Artist"
                )
                visualizations['top_artists_bar'] = fig_artists.to_json()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {}
    
    def get_comprehensive_dashboard(
        self,
        time_range: str = 'medium_term'
    ) -> Dict:
        """Generate comprehensive analytics dashboard."""
        try:
            # Gather all data
            top_tracks = self.get_top_tracks(time_range=time_range, limit=50)
            top_artists = self.get_top_artists(time_range=time_range, limit=20)
            recent_tracks = self.get_recently_played(limit=50)
            
            # Analyze patterns
            listening_analysis = self.analyze_listening_patterns(top_tracks)
            
            # Perform association mining
            association_results = self.perform_association_mining(
                recent_tracks,
                min_support=0.15,
                min_confidence=0.6
            )
            
            # Generate visualizations
            visualizations = self.generate_visualizations(listening_analysis)
            
            # Compile dashboard
            dashboard = {
                'overview': {
                    'time_range': time_range,
                    'total_top_tracks': len(top_tracks),
                    'total_top_artists': len(top_artists),
                    'recent_tracks_analyzed': len(recent_tracks)
                },
                'listening_patterns': listening_analysis,
                'top_tracks': top_tracks[:10],
                'top_artists': top_artists[:10],
                'associations': association_results,
                'visualizations': visualizations,
                'recommendations': self._generate_recommendations(
                    listening_analysis,
                    association_results
                )
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}")
            raise
    
    def _generate_recommendations(
        self,
        analysis: Dict,
        associations: Dict
    ) -> List[str]:
        """Generate personalized recommendations based on analysis."""
        recommendations = []
        
        # Based on audio features
        if 'audio_features' in analysis:
            features = analysis['audio_features']
            
            valence = features.get('valence')
            energy = features.get('energy')
            acousticness = features.get('acousticness')
            
            if valence is not None and valence > 0.6:
                recommendations.append(
                    "You love uplifting music! Try exploring more feel-good playlists."
                )
            if energy is not None and energy > 0.7:
                recommendations.append(
                    "High energy listener detected! Check out workout and party playlists."
                )
            if acousticness is not None and acousticness > 0.5:
                recommendations.append(
                    "You appreciate acoustic vibes. Explore unplugged and live sessions."
                )
        
        # Based on associations
        if associations and 'artist_associations' in associations:
            if associations['artist_associations']:
                recommendations.append(
                    "We found patterns in your listening! Artists you love often pair well together."
                )
        
        return recommendations if recommendations else [
            "Keep exploring! We'll provide better insights as you listen to more music."
        ]
