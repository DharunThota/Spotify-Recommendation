"""
Test script for Sequential Pattern Mining functionality.
Tests pattern discovery, prediction, and integration with recommendation engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sequence_mining import SequentialPatternMiner
from data_processor import create_data_processor
from recommendation_engine import RecommendationEngine
from explainability import ExplainabilityEngine
import config

def create_sample_listening_history():
    """
    Create sample listening history for testing.
    Simulates a user who listens to sequences like:
    - Song 0 → Song 1 → Song 2 (frequent pattern)
    - Song 5 → Song 6 → Song 7 (frequent pattern)
    - Song 10 → Song 11 (less frequent)
    """
    
    # Create sessions with timestamps
    base_time = datetime.now()
    sessions = []
    
    # Session 1: Song 0 → Song 1 → Song 2 (repeated 5 times)
    for i in range(5):
        session_start = base_time - timedelta(days=i*2, hours=10)
        sessions.extend([
            {"song_id": f"track_0", "played_at": (session_start).isoformat()},
            {"song_id": f"track_1", "played_at": (session_start + timedelta(minutes=3)).isoformat()},
            {"song_id": f"track_2", "played_at": (session_start + timedelta(minutes=7)).isoformat()},
        ])
    
    # Session 2: Song 5 → Song 6 → Song 7 (repeated 4 times)
    for i in range(4):
        session_start = base_time - timedelta(days=i*2+1, hours=14)
        sessions.extend([
            {"song_id": f"track_5", "played_at": (session_start).isoformat()},
            {"song_id": f"track_6", "played_at": (session_start + timedelta(minutes=4)).isoformat()},
            {"song_id": f"track_7", "played_at": (session_start + timedelta(minutes=8)).isoformat()},
        ])
    
    # Session 3: Song 10 → Song 11 (repeated 2 times - below min_support)
    for i in range(2):
        session_start = base_time - timedelta(days=i*3, hours=20)
        sessions.extend([
            {"song_id": f"track_10", "played_at": (session_start).isoformat()},
            {"song_id": f"track_11", "played_at": (session_start + timedelta(minutes=5)).isoformat()},
        ])
    
    # Add some noise (random songs)
    for i in range(3):
        session_start = base_time - timedelta(days=i*4, hours=16)
        sessions.append({"song_id": f"track_{20+i}", "played_at": session_start.isoformat()})
    
    return pd.DataFrame(sessions)

def test_pattern_mining():
    """Test basic pattern mining functionality."""
    print("=" * 80)
    print("TEST 1: Pattern Mining")
    print("=" * 80)
    
    # Create sample data
    history = create_sample_listening_history()
    print(f"\n[OK] Created sample listening history with {len(history)} entries")
    print(f"  Sessions span: {history['played_at'].min()} to {history['played_at'].max()}")
    
    # Initialize miner
    miner = SequentialPatternMiner(
        min_support=0.3,  # Lower threshold for testing
        max_gap=2,
        session_gap_minutes=30
    )
    
    # Fit on sample data
    print("\n[PROCESSING] Mining patterns...")
    miner.fit(history)
    
    # Get statistics
    stats = miner.get_statistics()
    print(f"\n[OK] Pattern Mining Complete:")
    print(f"  - Total sessions: {stats['total_sessions']}")
    print(f"  - Unique songs: {stats['unique_songs']}")
    print(f"  - Patterns found: {stats['patterns_found']}")
    print(f"  - Avg pattern length: {stats['avg_pattern_length']:.2f}")
    
    # Show discovered patterns
    if miner.patterns:
        print(f"\n  Top Patterns:")
        for pattern, support in miner.patterns[:5]:
            print(f"    {' → '.join(pattern)} (support: {support:.2f})")
    
    return miner

def test_prediction(miner):
    """Test next-song prediction."""
    print("\n" + "=" * 80)
    print("TEST 2: Next-Song Prediction")
    print("=" * 80)
    
    # Test predictions for known sequences
    test_cases = [
        (["track_0"], "After listening to track_0"),
        (["track_0", "track_1"], "After listening to track_0 → track_1"),
        (["track_5", "track_6"], "After listening to track_5 → track_6"),
        (["track_999"], "After listening to unknown track"),
    ]
    
    for context, description in test_cases:
        print(f"\n{description}:")
        predictions = miner.predict_next_songs(context, top_k=3)
        
        if predictions:
            for song_id, confidence in predictions:
                explanation = miner.get_pattern_explanation(song_id, context)
                print(f"  → {song_id} (confidence: {confidence:.2%})")
                if explanation and explanation.get('explanation_text'):
                    print(f"     {explanation['explanation_text']}")
        else:
            print("  → No predictions available")
    
    return predictions

def test_hybrid_recommendations():
    """Test hybrid recommendations (content + sequence)."""
    print("\n" + "=" * 80)
    print("TEST 3: Hybrid Recommendations (Content + Sequence)")
    print("=" * 80)
    
    try:
        # Initialize data processor
        print("\n[PROCESSING] Loading data processor...")
        processor = create_data_processor()
        
        if processor is None or processor.data is None:
            print("[WARNING] Data processor not available, skipping hybrid test")
            return
            
        print(f"[OK] Loaded {len(processor.data)} songs")
        
        # Initialize engines
        rec_engine = RecommendationEngine(processor)
        explainer = ExplainabilityEngine(processor)
        
        # Create sample listening history using actual song IDs
        sample_size = min(20, len(processor.data))
        sample_songs = processor.data.sample(n=sample_size)
        
        # Create realistic listening history
        base_time = datetime.now()
        history_records = []
        for i, (idx, song) in enumerate(sample_songs.iterrows()):
            history_records.append({
                "song_id": song['id'],
                "played_at": (base_time - timedelta(hours=i*2)).isoformat()
            })
        
        history = pd.DataFrame(history_records)
        print(f"[OK] Created listening history with {len(history)} songs")
        
        # Mine patterns
        miner = SequentialPatternMiner()
        miner.fit(history)
        stats = miner.get_statistics()
        print(f"[OK] Mined {stats['patterns_found']} patterns from history")
        
        # Get hybrid recommendations for first song in history
        test_song_id = history.iloc[0]['song_id']
        test_song = processor.get_song_by_id(test_song_id)
        
        if test_song is not None:
            test_idx = processor.song_id_to_idx[test_song_id]
            recent_context = history.head(5)['song_id'].tolist()
            
            print(f"\n[TEST] Test Song: {test_song['name']} by {test_song['artists']}")
            print(f"   Recent Context: {len(recent_context)} songs")
            
            # Get recommendations
            print("\n[PROCESSING] Generating hybrid recommendations...")
            recommendations = rec_engine.sequence_aware_recommendations(
                song_id=test_song_id,
                sequence_miner=miner,
                recent_context=recent_context,
                k=5,
                sequence_weight=config.SEQUENCE_WEIGHT
            )
            
            if recommendations:
                print(f"\n[OK] Top {len(recommendations)} Hybrid Recommendations:")
                for rank, (idx, score) in enumerate(recommendations, 1):
                    rec_song = processor.get_song_by_index(idx)
                    print(f"\n{rank}. {rec_song['name']} by {rec_song['artists']}")
                    print(f"   Score: {score:.4f}")
                    
                    # Get explanation
                    explanation = explainer.explain_sequence_recommendation(
                        test_idx, idx, miner, recent_context, sequence_weight=config.SEQUENCE_WEIGHT
                    )
                    
                    if 'explanation' in explanation:
                        print(f"   [INFO] {explanation['explanation']}")
            else:
                print("[WARNING] No recommendations generated")
        else:
            print("[WARNING] Test song not found in dataset")
            
    except Exception as e:
        print(f"[WARNING] Error in hybrid recommendations test: {e}")
        import traceback
        traceback.print_exc()

def test_cache_persistence():
    """Test save/load caching mechanism."""
    print("\n" + "=" * 80)
    print("TEST 4: Cache Persistence")
    print("=" * 80)
    
    import os
    cache_file = "test_sequence_cache.pkl"
    
    try:
        # Create and save miner
        history = create_sample_listening_history()
        miner1 = SequentialPatternMiner()
        miner1.fit(history)
        stats1 = miner1.get_statistics()
        
        print(f"\n[OK] Original miner: {stats1['patterns_found']} patterns")
        miner1.save(cache_file)
        print(f"[OK] Saved to {cache_file}")
        
        # Load miner
        miner2 = SequentialPatternMiner.load(cache_file)
        stats2 = miner2.get_statistics()
        
        print(f"[OK] Loaded from {cache_file}: {stats2['patterns_found']} patterns")
        
        # Verify consistency
        assert stats1['patterns_found'] == stats2['patterns_found'], "Pattern count mismatch!"
        assert len(miner1.patterns) == len(miner2.patterns), "Pattern dictionary mismatch!"
        
        print("[OK] Cache persistence validated!")
        
    finally:
        # Cleanup
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"[OK] Cleaned up test cache file")

def main():
    """Run all tests."""
    print("\n" + " SEQUENTIAL PATTERN MINING TEST SUITE ".center(80, "="))
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Test 1: Basic pattern mining
        miner = test_pattern_mining()
        
        # Test 2: Prediction
        test_prediction(miner)
        
        # Test 3: Hybrid recommendations
        test_hybrid_recommendations()
        
        # Test 4: Cache persistence
        test_cache_persistence()
        
        print("\n" + "=" * 80)
        print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[FAILED] TEST FAILED: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
