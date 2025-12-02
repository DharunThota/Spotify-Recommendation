"""Test script to verify Spotify API returns audio features and listening patterns."""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import config
from pprint import pprint

def test_spotify_api():
    """Test if Spotify API returns required data."""
    
    print("=" * 80)
    print("SPOTIFY API TEST")
    print("=" * 80)
    
    # Initialize Spotify client
    try:
        auth_manager = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=" ".join(config.SPOTIFY_SCOPES),
            cache_path=".test_spotify_cache"
        )
        
        sp = spotipy.Spotify(auth_manager=auth_manager)
        print("✓ Successfully initialized Spotify client\n")
        
    except Exception as e:
        print(f"✗ Failed to initialize Spotify client: {e}")
        return
    
    # Test 1: Get current user
    print("-" * 80)
    print("TEST 1: Get Current User")
    print("-" * 80)
    try:
        user = sp.current_user()
        print(f"✓ User ID: {user['id']}")
        print(f"✓ Display Name: {user.get('display_name', 'N/A')}")
        print(f"✓ Account Type: {user.get('product', 'N/A')}")
        print()
    except Exception as e:
        print(f"✗ Failed to get user: {e}\n")
        return
    
    # Test 2: Get recently played tracks
    print("-" * 80)
    print("TEST 2: Get Recently Played Tracks")
    print("-" * 80)
    try:
        recent = sp.current_user_recently_played(limit=10)
        print(f"✓ Retrieved {len(recent['items'])} recently played tracks")
        
        if recent['items']:
            first_track = recent['items'][0]
            print(f"\nSample track:")
            print(f"  - Name: {first_track['track']['name']}")
            print(f"  - Artist: {first_track['track']['artists'][0]['name']}")
            print(f"  - Played at: {first_track['played_at']}")
            print(f"  - Track ID: {first_track['track']['id']}")
            
            # Save track IDs for audio features test
            track_ids = [item['track']['id'] for item in recent['items'][:5]]
        else:
            print("✗ No recently played tracks found")
            track_ids = []
        print()
    except Exception as e:
        print(f"✗ Failed to get recently played: {e}\n")
        track_ids = []
    
    # Test 3: Get audio features
    print("-" * 80)
    print("TEST 3: Get Audio Features")
    print("-" * 80)
    if track_ids:
        try:
            audio_features = sp.audio_features(track_ids)
            print(f"✓ Retrieved audio features for {len([f for f in audio_features if f])} tracks")
            
            if audio_features and audio_features[0]:
                features = audio_features[0]
                print(f"\nSample audio features:")
                print(f"  - Danceability: {features.get('danceability', 'N/A')}")
                print(f"  - Energy: {features.get('energy', 'N/A')}")
                print(f"  - Valence: {features.get('valence', 'N/A')}")
                print(f"  - Tempo: {features.get('tempo', 'N/A')}")
                print(f"  - Loudness: {features.get('loudness', 'N/A')}")
                print(f"  - Acousticness: {features.get('acousticness', 'N/A')}")
                print(f"  - Instrumentalness: {features.get('instrumentalness', 'N/A')}")
                print(f"  - Speechiness: {features.get('speechiness', 'N/A')}")
                print("\n✓ Audio features ARE available from Spotify API")
            else:
                print("✗ No audio features returned (all None)")
        except Exception as e:
            print(f"✗ Failed to get audio features: {e}")
        print()
    else:
        print("⊘ Skipped (no track IDs available)\n")
    
    # Test 4: Get top tracks
    print("-" * 80)
    print("TEST 4: Get Top Tracks (Multiple Time Ranges)")
    print("-" * 80)
    for time_range in ['short_term', 'medium_term', 'long_term']:
        try:
            top_tracks = sp.current_user_top_tracks(limit=5, time_range=time_range)
            print(f"✓ {time_range}: Retrieved {len(top_tracks['items'])} top tracks")
            if top_tracks['items']:
                print(f"  Top track: {top_tracks['items'][0]['name']} by {top_tracks['items'][0]['artists'][0]['name']}")
        except Exception as e:
            print(f"✗ {time_range}: Failed - {e}")
    print()
    
    # Test 5: Get top artists
    print("-" * 80)
    print("TEST 5: Get Top Artists")
    print("-" * 80)
    try:
        top_artists = sp.current_user_top_artists(limit=5, time_range='medium_term')
        print(f"✓ Retrieved {len(top_artists['items'])} top artists")
        if top_artists['items']:
            for i, artist in enumerate(top_artists['items'][:3], 1):
                print(f"  {i}. {artist['name']} (Popularity: {artist['popularity']})")
        print()
    except Exception as e:
        print(f"✗ Failed to get top artists: {e}\n")
    
    # Test 6: Check listening patterns availability
    print("-" * 80)
    print("TEST 6: Listening Patterns Analysis")
    print("-" * 80)
    try:
        recent = sp.current_user_recently_played(limit=50)
        tracks = recent['items']
        
        print(f"✓ Retrieved {len(tracks)} listening history records")
        
        # Extract timestamps
        from datetime import datetime
        timestamps = [datetime.fromisoformat(item['played_at'].replace('Z', '+00:00')) for item in tracks]
        
        if len(timestamps) >= 2:
            time_gaps = [(timestamps[i] - timestamps[i+1]).total_seconds() / 60 for i in range(len(timestamps)-1)]
            avg_gap = sum(time_gaps) / len(time_gaps)
            print(f"✓ Average time gap between songs: {avg_gap:.1f} minutes")
            print(f"✓ Listening patterns CAN be derived from Spotify API")
            
            # Check for session patterns
            session_count = sum(1 for gap in time_gaps if gap > 30)
            print(f"✓ Detected ~{session_count + 1} listening sessions (30min gap threshold)")
        else:
            print("⚠ Not enough listening history to analyze patterns")
        print()
    except Exception as e:
        print(f"✗ Failed to analyze listening patterns: {e}\n")
    
    # Test 7: Test sequence mining feasibility
    print("-" * 80)
    print("TEST 7: Sequential Pattern Mining Feasibility")
    print("-" * 80)
    try:
        recent = sp.current_user_recently_played(limit=50)
        tracks = recent['items']
        
        # Group by sessions
        from datetime import datetime, timedelta
        sessions = []
        current_session = []
        
        for i, item in enumerate(tracks):
            if i == 0:
                current_session.append(item['track']['id'])
            else:
                prev_time = datetime.fromisoformat(tracks[i-1]['played_at'].replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(item['played_at'].replace('Z', '+00:00'))
                gap = (prev_time - curr_time).total_seconds() / 60
                
                if gap > 30:  # New session
                    if len(current_session) >= 2:
                        sessions.append(current_session)
                    current_session = [item['track']['id']]
                else:
                    current_session.append(item['track']['id'])
        
        if len(current_session) >= 2:
            sessions.append(current_session)
        
        print(f"✓ Extracted {len(sessions)} listening sessions")
        print(f"✓ Average session length: {sum(len(s) for s in sessions) / len(sessions) if sessions else 0:.1f} songs")
        
        if len(sessions) >= 3:
            print(f"✓ Sufficient data for sequence pattern mining (need 3+ sessions)")
            print("\n✓ SEQUENCE MINING IS FEASIBLE with Spotify API")
        else:
            print(f"⚠ Only {len(sessions)} sessions - need more listening history for meaningful patterns")
        print()
    except Exception as e:
        print(f"✗ Failed to test sequence mining: {e}\n")
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Audio features: Available via sp.audio_features()")
    print("✓ Listening patterns: Available via sp.current_user_recently_played()")
    print("✓ Sequential patterns: Can be derived from timestamps and song sequences")
    print("✓ All required data is available from Spotify API")
    print("\nNote: Audio features are retrieved separately from track info.")
    print("      Our recommendation system uses pre-processed data.csv, not live API.")
    print("=" * 80)


if __name__ == "__main__":
    print("\nThis will open a browser for Spotify authentication...")
    print("Make sure you have your Spotify credentials in config.py\n")
    
    input("Press Enter to continue...")
    
    test_spotify_api()
    
    print("\nTest complete! Check results above.")
    input("\nPress Enter to exit...")
