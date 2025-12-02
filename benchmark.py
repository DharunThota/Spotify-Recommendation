"""
Comparison script to benchmark Pandas vs PySpark performance.
Run this to see which engine works best for your dataset size.
"""

import time
import config
from data_processor import create_data_processor

def benchmark_processor(use_pyspark: bool):
    """Benchmark a data processor implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {'PySpark' if use_pyspark else 'Pandas'}")
    print(f"{'='*60}\n")
    
    # Update config
    original_setting = config.USE_PYSPARK
    config.USE_PYSPARK = use_pyspark
    
    try:
        # Create processor
        start_time = time.time()
        processor = create_data_processor()
        create_time = time.time() - start_time
        print(f"✓ Processor created: {create_time:.2f}s")
        
        # Initialize (load data)
        start_time = time.time()
        processor.initialize(force_reload=False)  # Use cache if available
        init_time = time.time() - start_time
        print(f"✓ Data loaded and processed: {init_time:.2f}s")
        
        # Test search
        start_time = time.time()
        results = processor.search_songs("love", limit=20)
        search_time = time.time() - start_time
        print(f"✓ Search completed: {search_time:.3f}s ({len(results)} results)")
        
        # Test mood extraction
        start_time = time.time()
        mood_songs = processor.get_songs_by_mood("happy", limit=50)
        mood_time = time.time() - start_time
        print(f"✓ Mood filtering: {mood_time:.3f}s ({len(mood_songs)} songs)")
        
        # Get dataset size
        if hasattr(processor, 'data_pandas') and processor.data_pandas is not None:
            total_songs = int(len(processor.data_pandas))
        elif hasattr(processor, 'data') and processor.data is not None:
            # For regular pandas DataFrame
            if hasattr(processor.data, '__len__'):
                total_songs = int(len(processor.data))
            # For PySpark DataFrame, use count()
            elif hasattr(processor.data, 'count'):
                total_songs = int(processor.data.count())
            else:
                total_songs = 0
        else:
            total_songs = 0
        
        print(f"\nDataset: {total_songs:,} songs")
        print(f"Total time: {create_time + init_time:.2f}s")
        
        # Cleanup
        if use_pyspark and hasattr(processor, 'stop_spark'):
            processor.stop_spark()
        
        return {
            'engine': 'PySpark' if use_pyspark else 'Pandas',
            'create_time': create_time,
            'init_time': init_time,
            'search_time': search_time,
            'mood_time': mood_time,
            'total_time': create_time + init_time,
            'dataset_size': total_songs
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    finally:
        # Restore original setting
        config.USE_PYSPARK = original_setting


def main():
    """Run benchmarks for both engines."""
    print("\n" + "="*60)
    print("Spotify Recommendation System - Performance Comparison")
    print("="*60)
    
    results = []
    
    # Benchmark Pandas
    pandas_result = benchmark_processor(use_pyspark=False)
    if pandas_result:
        results.append(pandas_result)
    
    # Benchmark PySpark
    try:
        pyspark_result = benchmark_processor(use_pyspark=True)
        if pyspark_result:
            results.append(pyspark_result)
    except ImportError:
        print("\n✗ PySpark not installed. Install with: pip install pyspark")
    
    # Display comparison
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nDataset Size: {results[0]['dataset_size']:,} songs\n")
        
        print(f"{'Metric':<25} {'Pandas':<15} {'PySpark':<15} {'Winner':<10}")
        print("-" * 65)
        
        metrics = [
            ('Creation Time', 'create_time'),
            ('Initialization Time', 'init_time'),
            ('Search Time', 'search_time'),
            ('Mood Filter Time', 'mood_time'),
            ('Total Time', 'total_time')
        ]
        
        for label, key in metrics:
            pandas_val = results[0][key]
            pyspark_val = results[1][key]
            winner = 'Pandas' if pandas_val < pyspark_val else 'PySpark'
            
            print(f"{label:<25} {pandas_val:<15.3f} {pyspark_val:<15.3f} {winner:<10}")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        dataset_size = results[0]['dataset_size']
        
        if dataset_size < 100000:
            print("\n✓ For your dataset size (<100K songs), Pandas is recommended.")
            print("  - Faster initialization and lower overhead")
            print("  - Simpler setup and debugging")
        elif dataset_size < 1000000:
            print("\n⚖ For your dataset size (100K-1M songs), both work well.")
            print("  - Pandas: Better for single-machine workloads")
            print("  - PySpark: Better if planning to scale further")
        else:
            print("\n✓ For your dataset size (>1M songs), PySpark is recommended.")
            print("  - Better memory management")
            print("  - Scales to distributed clusters")
        
        print("\nTo switch engines, edit config.py:")
        print("  USE_PYSPARK = True   # Use PySpark")
        print("  USE_PYSPARK = False  # Use Pandas")
        
    elif len(results) == 1:
        print("\n✓ Benchmarked Pandas only")
        print("  Install PySpark to compare: pip install pyspark")


if __name__ == "__main__":
    main()
