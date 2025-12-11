#!/bin/bash
# Startup script for the Agent Web Application

cd "$(dirname "$0")"

echo "Starting Agent Web Application..."
echo "Checking dependencies..."

# Check if Python can import required modules
python3 -c "import gradio, langchain, chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may not be installed."
    echo "Please ensure you have installed: gradio, langchain, chromadb"
    echo ""
fi

echo "Launching web interface..."
python3 agent_web_app.py

