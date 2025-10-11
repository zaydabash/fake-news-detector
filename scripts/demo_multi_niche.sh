#!/bin/bash

# Multi-Niche Fake News Detector Demo Script
# This script demonstrates the complete multi-niche system

set -e

echo "🚀 Multi-Niche Fake News Detector Demo"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'make setup' first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "🔧 Step 1: Demonstrate Data Collection for Climate Claims"
echo "--------------------------------------------------------"
echo "Collecting data for climate_claims niche..."
python -m src.data.pipeline --niche climate_claims

echo ""
echo "🤖 Step 2: Train Climate Claims Model"
echo "------------------------------------"
echo "Training model for climate_claims..."
python -m src.models.train_niche --niche climate_claims

echo ""
echo "📊 Step 3: Evaluate Climate Claims Model"
echo "---------------------------------------"
echo "Evaluating climate_claims model..."
python -m src.models.eval_niche --niche climate_claims

echo ""
echo "🎯 Step 4: Test Predictions"
echo "--------------------------"
echo "Testing climate claims prediction:"
python -m src.predict.cli predict-text "Climate change is a hoax perpetrated by scientists to get funding"

echo ""
echo "Testing celebrity rumors prediction:"
python -m src.predict.cli predict-text "Breaking: Celebrity spotted with mystery person at exclusive restaurant"

echo ""
echo "Testing hustle scams prediction:"
python -m src.predict.cli predict-text "Make $10,000 per day with this secret method! DM me for details!"

echo ""
echo "🌐 Step 5: Launch Web Interface"
echo "-------------------------------"
echo "Starting Streamlit demo with niche selection..."
echo "The web interface will open in your browser."
echo "You can select different niches from the dropdown menu."
echo ""
echo "Press Ctrl+C to stop the demo when you're done exploring."
echo ""

streamlit run app/streamlit_app.py

echo ""
echo "✅ Demo Complete!"
echo "================"
echo ""
echo "What you've seen:"
echo "  🌍 Multi-niche data collection and weak labeling"
echo "  🤖 Per-niche model training and evaluation"
echo "  🎯 Real-time predictions across different niches"
echo "  🌐 Interactive web interface with niche selection"
echo ""
echo "Next steps:"
echo "  📚 Explore other niches: make data NICHE=celebrity_rumors"
echo "  🔄 Train unified model: make unified-train"
echo "  📊 View evaluation reports in artifacts/<niche>/reports/"
echo "  🧪 Try different text samples in the web interface"
echo ""
echo "🎉 Multi-Niche Fake News Detector is ready for production use!"
