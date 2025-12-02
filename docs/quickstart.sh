#!/bin/bash
# Quick Start Script for Spotify Recommendation System

echo "=================================="
echo "Spotify Recommendation System"
echo "Quick Start Guide"
echo "=================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed!"
    exit 1
fi
echo "✅ Python is installed"
echo ""

# Install dependencies
echo "2. Installing dependencies..."
read -p "Install/update dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
fi
echo ""

# Check PySpark
echo "3. Checking PySpark availability..."
python3 -c "import pyspark; print('✅ PySpark', pyspark.__version__, 'is installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  PySpark not installed (optional)"
    read -p "Install PySpark for big data processing? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install pyspark
        echo "✅ PySpark installed"
    else
        echo "ℹ️  Continuing with Pandas (recommended for datasets < 1M songs)"
    fi
fi
echo ""

# Check data directory
echo "4. Checking data files..."
if [ ! -d "data" ]; then
    echo "❌ 'data' directory not found!"
    exit 1
fi

if [ ! -f "data/data.csv" ]; then
    echo "❌ 'data/data.csv' not found!"
    exit 1
fi
echo "✅ Data files found"
echo ""

# Process data
echo "5. Processing data..."
if [ ! -f "processed_data.pkl" ] && [ ! -f "processed_data_pyspark.pkl" ]; then
    echo "ℹ️  No processed data found. Running data processor..."
    read -p "Process data now? This may take a few minutes. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 data_processor.py
        if [ $? -ne 0 ]; then
            echo "❌ Data processing failed"
            exit 1
        fi
        echo "✅ Data processed successfully"
    else
        echo "⚠️  Skipping data processing. First run will take longer."
    fi
else
    echo "✅ Processed data already exists"
fi
echo ""

# Run benchmark (optional)
echo "6. Performance benchmark (optional)..."
read -p "Run performance benchmark? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 benchmark.py
fi
echo ""

# Show configuration
echo "7. Current configuration:"
python3 -c "import config; print(f'   Processing Engine: {'PySpark' if config.USE_PYSPARK else 'Pandas'}'); print(f'   Clusters: {config.N_CLUSTERS}'); print(f'   Recommendations: {config.N_RECOMMENDATIONS}')"
echo ""

# Start server
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Start the server with:"
echo "   python3 main.py"
echo ""
echo "Or with uvicorn:"
echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit:"
echo "   🌐 Web Interface: http://localhost:8000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo "   ❤️  Health Check: http://localhost:8000/api/health"
echo ""
echo "Configuration:"
echo "   📝 Edit config.py to switch between Pandas and PySpark"
echo "   📖 Read PYSPARK_GUIDE.md for PySpark setup"
echo "   📋 Read CONFIG_EXAMPLES.md for configuration templates"
echo ""
