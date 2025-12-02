# Architecture Overview - PySpark Integration

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Web Browser / API Client)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Application                        │
│                          (main.py)                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Search     │  │Recommendations│  │Explainability│         │
│  │  Endpoints   │  │   Endpoints   │  │  Endpoints   │         │
│  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘         │
│         │                  │                   │                 │
│         └──────────────────┴───────────────────┘                │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ RecommendationEngine│ │Explainability│ │   Other         │
│(recommendation  │ │    Engine       │ │   Modules       │
│ _engine.py)     │ │(explainability  │ │                 │
│                 │ │     .py)        │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             │ Uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Data Processor Factory                          │
│                 (create_data_processor())                        │
│                                                                  │
│                  Reads: config.USE_PYSPARK                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
     USE_PYSPARK = False       USE_PYSPARK = True
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    DataProcessor         │  │  PySparkDataProcessor    │
│    (Pandas-based)        │  │    (PySpark-based)       │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Load CSV           │  │  │  │ Spark Session      │  │
│  │ (pandas.read_csv)  │  │  │  │ Management         │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Data Cleaning      │  │  │  │ Load CSV           │  │
│  │ (pandas ops)       │  │  │  │ (spark.read.csv)   │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Normalization      │  │  │  │ Data Cleaning      │  │
│  │ (sklearn)          │  │  │  │ (PySpark SQL)      │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ K-Means            │  │  │  │ Normalization      │  │
│  │ (sklearn)          │  │  │  │ (PySpark ML)       │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Mood Extraction    │  │  │  │ K-Means            │  │
│  │                    │  │  │  │ (PySpark ML)       │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Indexing           │  │  │  │ Mood Extraction    │  │
│  │                    │  │  │  │                    │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │ Save/Load          │  │  │  │ Indexing           │  │
│  │ (pickle)           │  │  │  │                    │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│                          │  │                          │
│                          │  │  ┌────────────────────┐  │
│                          │  │  │ Convert to Pandas  │  │
│                          │  │  │ (for compatibility)│  │
│                          │  │  └────────────────────┘  │
│                          │  │                          │
│                          │  │  ┌────────────────────┐  │
│                          │  │  │ Save/Load          │  │
│                          │  │  │ (pickle)           │  │
│                          │  │  └────────────────────┘  │
└────────┬─────────────────┘  └────────┬─────────────────┘
         │                              │
         │                              ▼
         │                    ┌───────────────────┐
         │                    │   Spark Cluster   │
         │                    │   (Optional)      │
         │                    │                   │
         │                    │ ┌───┐ ┌───┐ ┌───┐│
         │                    │ │ E │ │ E │ │ E ││
         │                    │ │ x │ │ x │ │ x ││
         │                    │ │ e │ │ e │ │ e ││
         │                    │ └───┘ └───┘ └───┘│
         │                    └───────────────────┘
         │                              
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │        Data Storage          │
         │                              │
         │  ┌────────────────────────┐  │
         │  │  CSV Files (data/*.csv)│  │
         │  └────────────────────────┘  │
         │                              │
         │  ┌────────────────────────┐  │
         │  │  Processed Data        │  │
         │  │  - processed_data.pkl  │  │
         │  │  - processed_data_     │  │
         │  │    pyspark.pkl         │  │
         │  └────────────────────────┘  │
         └──────────────────────────────┘
```

## Data Flow

### Initialization Flow

```
1. Application Starts (main.py)
   │
   ├─> Read config.USE_PYSPARK flag
   │
   ├─> Call create_data_processor()
   │   │
   │   ├─> If USE_PYSPARK = False
   │   │   └─> Create DataProcessor() [Pandas]
   │   │
   │   └─> If USE_PYSPARK = True
   │       └─> Create PySparkDataProcessor() [PySpark]
   │           └─> Initialize Spark Session
   │
   ├─> Call processor.initialize()
   │   │
   │   ├─> Check for cached pickle file
   │   │
   │   ├─> If cache exists:
   │   │   └─> Load from pickle (fast)
   │   │
   │   └─> If no cache:
   │       ├─> Load CSV files
   │       ├─> Clean data
   │       ├─> Create indexes
   │       ├─> Normalize features
   │       ├─> Extract mood features
   │       ├─> Perform K-Means clustering
   │       └─> Save to pickle
   │
   └─> Initialize RecommendationEngine & ExplainabilityEngine
```

### Recommendation Request Flow

```
1. API Request (/api/recommend/song)
   │
   ├─> FastAPI receives request
   │
   ├─> Validate song_id
   │
   ├─> RecommendationEngine.song_based_recommendations()
   │   │
   │   ├─> Get song index from processor
   │   ├─> Get feature vector from processor
   │   ├─> Find similar songs using cosine similarity
   │   ├─> Apply diversity filtering
   │   └─> Return top N recommendations
   │
   ├─> ExplainabilityEngine.explain_song_recommendation()
   │   │
   │   ├─> Analyze feature similarities
   │   ├─> Find common attributes
   │   └─> Generate explanation text
   │
   └─> Return JSON response with recommendations & explanations
```

## Component Responsibilities

### Configuration Layer (config.py)
- **Purpose**: Central configuration management
- **Key Settings**:
  - `USE_PYSPARK`: Toggle between engines
  - `PYSPARK_CONFIG`: Spark configuration
  - `AUDIO_FEATURES`: Feature columns
  - `MOOD_CRITERIA`: Mood definitions
  - Algorithm parameters

### Data Processing Layer (data_processor.py)
- **Purpose**: Data loading, cleaning, and feature engineering
- **Implementations**:
  1. **DataProcessor (Pandas)**:
     - Uses pandas for data manipulation
     - Uses sklearn for ML operations
     - In-memory processing
     - Best for < 1M songs
  
  2. **PySparkDataProcessor (PySpark)**:
     - Uses Spark DataFrames
     - Uses PySpark ML library
     - Distributed processing
     - Scalable to millions of songs
     - Maintains pandas compatibility

### Recommendation Layer (recommendation_engine.py)
- **Purpose**: Generate recommendations
- **Algorithms**:
  - Content-based filtering
  - Cosine similarity
  - Mood-based filtering
  - Hybrid recommendations
- **Engine Agnostic**: Works with both processors

### Explainability Layer (explainability.py)
- **Purpose**: Generate human-readable explanations
- **Features**:
  - Feature similarity analysis
  - Common attribute detection
  - Natural language generation
- **Engine Agnostic**: Works with both processors

### API Layer (main.py)
- **Purpose**: RESTful API interface
- **Framework**: FastAPI
- **Endpoints**:
  - Search songs
  - Song-based recommendations
  - Mood-based recommendations
  - Hybrid recommendations
  - Health check
- **Automatic Engine Selection**: Uses factory pattern

## Configuration Decision Tree

```
                    Start Application
                          │
                          ▼
              Read config.USE_PYSPARK
                          │
          ┌───────────────┴───────────────┐
          │                               │
    USE_PYSPARK = False          USE_PYSPARK = True
          │                               │
          ▼                               ▼
   Dataset < 1M songs?            Is Spark Installed?
          │                               │
    ┌─────┴─────┐                   ┌─────┴─────┐
   YES          NO                  YES          NO
    │            │                   │            │
    ▼            ▼                   ▼            ▼
Use Pandas   Consider            Use PySpark   Error!
(Fast)       PySpark             (Scalable)    Install
             (Scale)                           PySpark
```

## Scaling Strategies

### Vertical Scaling (Single Machine)
```
Small Dataset (< 100K)
    └─> Pandas + 4GB RAM

Medium Dataset (100K - 500K)
    └─> Pandas + 8GB RAM
    └─> Or PySpark local[*] + 8GB RAM

Large Dataset (500K - 2M)
    └─> PySpark local[*] + 16-32GB RAM

Very Large Dataset (> 2M)
    └─> PySpark local[*] + 32GB+ RAM
```

### Horizontal Scaling (Distributed)
```
Dataset > 10M songs
    └─> PySpark Cluster Mode
        ├─> Master Node (4-8GB)
        ├─> Worker Nodes (8-16GB each)
        └─> Scale workers as needed
```

## Key Design Decisions

1. **Factory Pattern**: Single entry point (`create_data_processor()`)
2. **Interface Consistency**: Both processors have identical methods
3. **Lazy Loading**: PySpark only imported when needed
4. **Pandas Compatibility**: PySpark converts to pandas for compatibility
5. **Separate Cache Files**: Different pickle files for each engine
6. **Engine-Agnostic Algorithms**: Recommendation logic independent of data engine
7. **Configuration-Driven**: Single flag to switch engines
8. **Graceful Degradation**: Falls back to Pandas if PySpark unavailable

## Performance Characteristics

| Operation | Pandas | PySpark (Local) | PySpark (Cluster) |
|-----------|--------|-----------------|-------------------|
| Load Data | Fast | Slower startup | Slower startup |
| Transform | Fast (< 1M) | Scales linearly | Parallel |
| Search | Fast | Similar | Faster |
| Clustering | Memory-bound | Partition-based | Distributed |
| Memory | All in RAM | Spill to disk | Distributed |

