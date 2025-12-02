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
import os
import config
from sequence_mining import SequentialPatternMiner

logger = logging.getLogger(__name__)


class SpotifyAnalytics:
    """Analytics engine for Spotify user listening history."""
    
    def __init__(self, cache_path=".spotify_cache"):
        """Initialize Spotify Analytics with OAuth."""
        self.cache_path = cache_path
        self.sp = None
        self.user_id = None
        self.sequence_miner = None
        
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
        """Get audio features for multiple tracks.
        
        Note: Spotify has deprecated the audio_features endpoint.
        This method now returns empty features with a warning.
        """
        if not self.sp:
            logger.error("Cannot fetch audio features: Not authenticated")
            raise Exception("Not authenticated. Please authenticate first.")
        
        logger.warning(f"Audio features endpoint has been deprecated by Spotify. Returning empty features for {len(track_ids)} tracks.")
        
        # Return empty features since Spotify deprecated the endpoint
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
    
    def calculate_artist_diversity_score(self, tracks: List[Dict]) -> Dict:
        """Calculate artist diversity score (unique artists / total songs)."""
        if not tracks:
            return {"score": 0, "classification": "No data"}
        
        df = pd.DataFrame(tracks)
        unique_artists = df['artist'].nunique()
        total_tracks = len(df)
        
        # Calculate diversity score (0-100)
        diversity_score = int((unique_artists / total_tracks) * 100)
        
        # Classification
        if diversity_score >= 75:
            classification = "Explorer"
            message = "You explore a wide variety of artists!"
        elif diversity_score >= 50:
            classification = "Balanced"
            message = "You have a healthy mix of favorites and new discoveries."
        else:
            classification = "Loyalist"
            message = "You're loyal to your favorite artists!"
        
        return {
            "score": diversity_score,
            "classification": classification,
            "message": message,
            "unique_artists": unique_artists,
            "total_tracks": total_tracks
        }
    
    def calculate_new_vs_old_taste(self, tracks: List[Dict]) -> Dict:
        """Analyze new vs old music taste based on release years."""
        if not self.sp or not tracks:
            return {"error": "No data available"}
        
        try:
            # Get full track details with album release dates
            track_ids = [t['id'] for t in tracks]
            release_years = []
            
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i:i+50]
                batch_tracks = self.sp.tracks(batch)['tracks']
                
                for track in batch_tracks:
                    if track and track.get('album'):
                        release_date = track['album'].get('release_date', '')
                        if release_date:
                            year = int(release_date.split('-')[0])
                            release_years.append(year)
            
            if not release_years:
                return {"error": "No release date information available"}
            
            current_year = datetime.now().year
            last_year = current_year - 1
            
            # Calculate percentages
            recent_count = sum(1 for year in release_years if year >= last_year)
            recent_percentage = int((recent_count / len(release_years)) * 100)
            
            # Average age of songs
            avg_age = current_year - np.mean(release_years)
            
            # Classification
            if recent_percentage >= 60:
                classification = "Trendsetter"
                message = f"You like to keep it fresh — {recent_percentage}% of your library is from recent releases"
            elif recent_percentage >= 30:
                classification = "Balanced"
                message = f"You enjoy a mix of new and classic — {recent_percentage}% recent releases"
            else:
                classification = "Nostalgic"
                message = f"You love the classics — only {recent_percentage}% from recent years"
            
            return {
                "recent_percentage": recent_percentage,
                "avg_age": round(avg_age, 1),
                "classification": classification,
                "message": message,
                "release_years": release_years
            }
        except Exception as e:
            logger.error(f"Error calculating new vs old taste: {str(e)}")
            return {"error": str(e)}
    
    def calculate_popularity_ranking(self, tracks: List[Dict]) -> Dict:
        """Calculate average popularity and classify listener type."""
        if not tracks:
            return {"score": 0, "classification": "No data"}
        
        df = pd.DataFrame(tracks)
        avg_popularity = df['popularity'].mean()
        
        # Classification
        if avg_popularity >= 80:
            classification = "Mainstream"
            message = "You love the hits! You're all about mainstream music."
            emoji = "🔥"
        elif avg_popularity >= 50:
            classification = "Balanced"
            message = "You enjoy a mix of popular and underground tracks."
            emoji = "🎵"
        else:
            classification = "Underground"
            message = f"You love discovering hidden gems — your avg popularity = {int(avg_popularity)}"
            emoji = "💎"
        
        return {
            "score": int(avg_popularity),
            "classification": classification,
            "message": message,
            "emoji": emoji
        }
    
    def analyze_playlist_personality(self, tracks: List[Dict]) -> Dict:
        """Analyze playlist characteristics."""
        if not tracks:
            return {}
        
        df = pd.DataFrame(tracks)
        
        # Calculate stats
        avg_duration_ms = df['duration_ms'].mean()
        avg_duration_min = int(avg_duration_ms / 60000)
        avg_duration_sec = int((avg_duration_ms % 60000) / 1000)
        
        # Most repeated artist
        top_artist = df['artist'].value_counts().head(1)
        most_common_artist = top_artist.index[0] if len(top_artist) > 0 else "Unknown"
        most_common_count = int(top_artist.values[0]) if len(top_artist) > 0 else 0
        
        # Most popular track
        most_popular_idx = df['popularity'].idxmax()
        most_popular_track = df.loc[most_popular_idx]
        
        return {
            "avg_duration": f"{avg_duration_min}m {avg_duration_sec}s",
            "avg_duration_ms": int(avg_duration_ms),
            "most_common_artist": most_common_artist,
            "most_common_artist_count": most_common_count,
            "most_popular_track": {
                "name": most_popular_track['name'],
                "artist": most_popular_track['artist'],
                "popularity": int(most_popular_track['popularity'])
            },
            "total_tracks": len(df)
        }
    
    def get_release_year_distribution(self, tracks: List[Dict]) -> Dict:
        """Get distribution of release years for trend analysis."""
        if not self.sp or not tracks:
            return {"error": "No data available"}
        
        try:
            track_ids = [t['id'] for t in tracks]
            year_counts = {}
            
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i:i+50]
                batch_tracks = self.sp.tracks(batch)['tracks']
                
                for track in batch_tracks:
                    if track and track.get('album'):
                        release_date = track['album'].get('release_date', '')
                        if release_date:
                            year = int(release_date.split('-')[0])
                            year_counts[year] = year_counts.get(year, 0) + 1
            
            if not year_counts:
                return {"error": "No release date information available"}
            
            # Find peak year
            peak_year = max(year_counts, key=year_counts.get)
            peak_count = year_counts[peak_year]
            
            # Convert to sorted list for visualization
            year_data = [{"year": year, "count": count} for year, count in sorted(year_counts.items())]
            
            return {
                "distribution": year_data,
                "peak_year": peak_year,
                "peak_count": peak_count,
                "message": f"Your music peaked in {peak_year} — nostalgia mode activated!" if peak_year < datetime.now().year - 3 else f"You're vibing with {peak_year} releases!"
            }
        except Exception as e:
            logger.error(f"Error getting release year distribution: {str(e)}")
            return {"error": str(e)}
    
    def determine_music_identity(self, wrapped_data: Dict) -> Dict:
        """Determine user's music identity based on all metrics."""
        # Collect scores from various analyses
        diversity_score = wrapped_data.get('artist_diversity', {}).get('score', 50)
        popularity_score = wrapped_data.get('popularity_ranking', {}).get('score', 50)
        recent_percentage = wrapped_data.get('new_vs_old', {}).get('recent_percentage', 50)
        
        # Determine primary identity
        identity_scores = {}
        
        # Explorer: High diversity, low popularity avg
        identity_scores['Explorer'] = (diversity_score * 0.5) + ((100 - popularity_score) * 0.3) + (recent_percentage * 0.2)
        
        # Superfan: Low diversity, high concentration
        identity_scores['Superfan'] = ((100 - diversity_score) * 0.6) + (popularity_score * 0.2) + ((100 - recent_percentage) * 0.2)
        
        # Trendsetter: Mostly new releases, high diversity
        identity_scores['Trendsetter'] = (recent_percentage * 0.5) + (diversity_score * 0.3) + ((100 - popularity_score) * 0.2)
        
        # Mainstream: High popularity, balanced diversity
        identity_scores['Mainstream'] = (popularity_score * 0.5) + (abs(diversity_score - 50) * -0.3) + (recent_percentage * 0.2)
        
        # Nostalgic: Old music preference
        identity_scores['Nostalgic'] = ((100 - recent_percentage) * 0.6) + (diversity_score * 0.2) + ((100 - popularity_score) * 0.2)
        
        # Find top identity
        primary_identity = max(identity_scores, key=identity_scores.get)
        primary_score = int(identity_scores[primary_identity])
        
        # Get secondary identity
        remaining = {k: v for k, v in identity_scores.items() if k != primary_identity}
        secondary_identity = max(remaining, key=remaining.get) if remaining else None
        
        # Identity descriptions
        descriptions = {
            'Explorer': "You're always discovering new artists and hidden gems across diverse genres.",
            'Superfan': "You're deeply devoted to your favorite artists and stick with what you love.",
            'Trendsetter': "You're ahead of the curve, always listening to the latest and freshest releases.",
            'Mainstream': "You love the hits and popular tracks that everyone's talking about.",
            'Nostalgic': "You cherish the classics and music from the past holds a special place in your heart."
        }
        
        return {
            "primary_identity": primary_identity,
            "primary_score": primary_score,
            "secondary_identity": secondary_identity,
            "description": descriptions.get(primary_identity, ""),
            "all_scores": identity_scores
        }
    
    def determine_music_identity(self, wrapped_data: Dict) -> Dict:
        """Determine user's music identity based on all metrics."""
        # Collect scores from various analyses
        diversity_score = wrapped_data.get('artist_diversity', {}).get('score', 50)
        popularity_score = wrapped_data.get('popularity_ranking', {}).get('score', 50)
        recent_percentage = wrapped_data.get('new_vs_old', {}).get('recent_percentage', 50)
        
        # Determine primary identity
        identity_scores = {}
        
        # Explorer: High diversity, low popularity avg
        identity_scores['Explorer'] = (diversity_score * 0.5) + ((100 - popularity_score) * 0.3) + (recent_percentage * 0.2)
        
        # Superfan: Low diversity, high concentration
        identity_scores['Superfan'] = ((100 - diversity_score) * 0.6) + (popularity_score * 0.2) + ((100 - recent_percentage) * 0.2)
        
        # Trendsetter: Mostly new releases, high diversity
        identity_scores['Trendsetter'] = (recent_percentage * 0.5) + (diversity_score * 0.3) + ((100 - popularity_score) * 0.2)
        
        # Mainstream: High popularity, balanced diversity
        identity_scores['Mainstream'] = (popularity_score * 0.5) + (abs(diversity_score - 50) * -0.3) + (recent_percentage * 0.2)
        
        # Nostalgic: Old music preference
        identity_scores['Nostalgic'] = ((100 - recent_percentage) * 0.6) + (diversity_score * 0.2) + ((100 - popularity_score) * 0.2)
        
        # Find top identity
        primary_identity = max(identity_scores, key=identity_scores.get)
        primary_score = int(identity_scores[primary_identity])
        
        # Get secondary identity
        remaining = {k: v for k, v in identity_scores.items() if k != primary_identity}
        secondary_identity = max(remaining, key=remaining.get) if remaining else None
        
        # Identity descriptions
        descriptions = {
            'Explorer': "You're always discovering new artists and hidden gems across diverse genres.",
            'Superfan': "You're deeply devoted to your favorite artists and stick with what you love.",
            'Trendsetter': "You're ahead of the curve, always listening to the latest and freshest releases.",
            'Mainstream': "You love the hits and popular tracks that everyone's talking about.",
            'Nostalgic': "You cherish the classics and music from the past holds a special place in your heart."
        }
        
        return {
            "primary_identity": primary_identity,
            "primary_score": primary_score,
            "secondary_identity": secondary_identity,
            "description": descriptions.get(primary_identity, ""),
            "all_scores": identity_scores
        }
    
    def analyze_listening_time_patterns(self) -> Dict:
        """Analyze listening patterns by time of day and day of week from recently played."""
        try:
            # Get recently played tracks (max 50)
            recent_tracks = self.get_recently_played(limit=50)
            
            if not recent_tracks:
                return {"error": "No recently played data available"}
            
            df = pd.DataFrame(recent_tracks)
            df['played_at'] = pd.to_datetime(df['played_at'])
            df['hour'] = df['played_at'].dt.hour
            df['day_of_week'] = df['played_at'].dt.day_name()
            df['date'] = df['played_at'].dt.date
            
            # Hour distribution
            hour_counts = df['hour'].value_counts().sort_index()
            hourly_data = [
                {
                    "hour": int(hour),
                    "count": int(count),
                    "period": "Night" if hour < 6 else "Morning" if hour < 12 else "Afternoon" if hour < 18 else "Evening"
                }
                for hour, count in hour_counts.items()
            ]
            
            # Day of week distribution
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['day_of_week'].value_counts()
            daily_data = [
                {"day": day, "count": int(day_counts.get(day, 0))}
                for day in day_order
            ]
            
            # Find peak listening times
            peak_hour = int(hour_counts.idxmax()) if not hour_counts.empty else 0
            peak_day = day_counts.idxmax() if not day_counts.empty else "Unknown"
            
            # Classify listener type based on peak times
            if peak_hour < 6:
                time_personality = "Night Owl"
                time_message = f"You love listening at {peak_hour}:00 - true night owl vibes!"
            elif peak_hour < 12:
                time_personality = "Early Bird"
                time_message = f"You start your day with music at {peak_hour}:00 - morning person energy!"
            elif peak_hour < 18:
                time_personality = "Afternoon Listener"
                time_message = f"Peak listening at {peak_hour}:00 - perfect afternoon soundtrack!"
            else:
                time_personality = "Evening Enthusiast"
                time_message = f"You unwind with music at {peak_hour}:00 - evening relaxation mode!"
            
            return {
                "hourly_data": hourly_data,
                "daily_data": daily_data,
                "peak_hour": peak_hour,
                "peak_day": peak_day,
                "time_personality": time_personality,
                "time_message": time_message,
                "total_plays": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing listening time patterns: {str(e)}")
            return {"error": str(e)}
    
    def get_wrapped_insights(self, time_range: str = 'medium_term') -> Dict:
        """Get comprehensive Spotify-Wrapped style insights."""
        try:
            # Gather data
            top_tracks = self.get_top_tracks(time_range=time_range, limit=50)
            top_artists = self.get_top_artists(time_range=time_range, limit=50)
            
            # Calculate all insights
            artist_diversity = self.calculate_artist_diversity_score(top_tracks)
            new_vs_old = self.calculate_new_vs_old_taste(top_tracks)
            popularity_ranking = self.calculate_popularity_ranking(top_tracks)
            playlist_personality = self.analyze_playlist_personality(top_tracks)
            release_trends = self.get_release_year_distribution(top_tracks)
            
            # Calculate top artists share percentage
            df_tracks = pd.DataFrame(top_tracks)
            artist_counts = df_tracks['artist'].value_counts()
            top_artists_with_share = []
            total_tracks = len(df_tracks)
            
            for idx, artist_data in enumerate(top_artists[:10]):
                artist_name = artist_data['name']
                count = artist_counts.get(artist_name, 0)
                share = int((count / total_tracks) * 100) if total_tracks > 0 else 0
                top_artists_with_share.append({
                    **artist_data,
                    'track_count': int(count),
                    'share_percentage': share
                })
            
            # Genre distribution
            genre_counts = {}
            for artist in top_artists:
                for genre in artist.get('genres', []):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            genre_distribution = [
                {"name": genre, "value": count, "percentage": int((count / len(top_artists)) * 100)}
                for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            ]
            
            # Time-based analysis
            time_patterns = self.analyze_listening_time_patterns()
            
            # Compile wrapped insights
            wrapped_data = {
                'artist_diversity': artist_diversity,
                'new_vs_old': new_vs_old,
                'popularity_ranking': popularity_ranking,
                'playlist_personality': playlist_personality,
                'release_trends': release_trends,
                'top_artists_with_share': top_artists_with_share,
                'genre_distribution': genre_distribution,
                'top_tracks': top_tracks[:10],
                'time_patterns': time_patterns
            }
            
            # Determine music identity
            music_identity = self.determine_music_identity(wrapped_data)
            wrapped_data['music_identity'] = music_identity
            
            return wrapped_data
            
        except Exception as e:
            logger.error(f"Error generating wrapped insights: {str(e)}")
            raise
    
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
    
    def get_listening_sequences(self, time_range: str = 'medium_term') -> List[List[str]]:
        """
        Extract listening sequences from recently played data.
        Groups songs by session (30-min gap = new session).
        
        Args:
            time_range: Time range for data ('short_term', 'medium_term', 'long_term')
            
        Returns:
            List of sequences where each sequence is list of song IDs
        """
        try:
            # Get recently played tracks (max 50 from API)
            recent_tracks = self.get_recently_played(limit=50)
            
            if not recent_tracks:
                logger.warning("No recently played tracks available")
                return []
            
            # SequentialPatternMiner will handle session extraction
            return recent_tracks
            
        except Exception as e:
            logger.error(f"Error extracting listening sequences: {str(e)}")
            return []
    
    def mine_listening_patterns(self, force_refresh: bool = False) -> Optional[SequentialPatternMiner]:
        """
        Mine sequential patterns from user's listening history.
        
        Args:
            force_refresh: If True, re-mine patterns even if cache exists
            
        Returns:
            SequentialPatternMiner instance with mined patterns
        """
        try:
            # Try to load from cache first
            if not force_refresh and os.path.exists(config.SEQUENCE_CACHE_FILE):
                logger.info(f"Loading sequence patterns from cache: {config.SEQUENCE_CACHE_FILE}")
                miner = SequentialPatternMiner(
                    min_support=config.SEQUENCE_MIN_SUPPORT,
                    max_gap=config.SEQUENCE_MAX_GAP,
                    session_gap_minutes=config.SEQUENCE_SESSION_GAP_MINUTES
                )
                miner.load(config.SEQUENCE_CACHE_FILE)
                self.sequence_miner = miner
                return miner
            
            # Get listening history
            logger.info("Mining new sequential patterns from listening history")
            listening_history = self.get_recently_played(limit=50)
            
            if not listening_history:
                logger.warning("No listening history available for pattern mining")
                return None
            
            # Initialize and fit miner
            miner = SequentialPatternMiner(
                min_support=config.SEQUENCE_MIN_SUPPORT,
                max_gap=config.SEQUENCE_MAX_GAP,
                session_gap_minutes=config.SEQUENCE_SESSION_GAP_MINUTES
            )
            
            miner.fit(listening_history)
            
            # Save to cache
            miner.save(config.SEQUENCE_CACHE_FILE)
            
            self.sequence_miner = miner
            return miner
            
        except Exception as e:
            logger.error(f"Error mining listening patterns: {str(e)}")
            return None
    
    def get_sequence_statistics(self) -> Dict:
        """Get statistics about mined sequential patterns."""
        if not self.sequence_miner:
            return {"error": "No patterns mined yet. Call mine_listening_patterns() first."}
        
        return self.sequence_miner.get_statistics()
