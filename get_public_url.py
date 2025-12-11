#!/usr/bin/env python3
"""
Helper script to find and open the Gradio public URL
"""
import subprocess
import re
import webbrowser
import time
import sys

def find_gradio_url():
    """Find the Gradio public URL from running processes"""
    try:
        # Check if app is running
        result = subprocess.run(['curl', '-s', 'http://localhost:7860'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("⚠️ App is not running on localhost:7860")
            return None
            
        # Try to get URL from Gradio's network connections or logs
        # When share=True, Gradio creates a tunnel and prints the URL
        # We can't easily extract it programmatically, but we can check stderr/stdout
        # of the running process
        
        # Check if there's a log file
        import os
        log_files = ['/tmp/gradio_app.log', 'gradio_output.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    urls = re.findall(r'https://[a-zA-Z0-9-]+\.gradio\.live', content)
                    if urls:
                        return urls[0]
        
        print("ℹ️  Public URL not found in logs.")
        print("The URL is printed when the app starts with share=True")
        print("\nTo see the public URL:")
        print("1. Check the terminal where you ran: python3 agent_web_app.py")
        print("2. Look for a line: 'Running on public URL: https://xxxxx.gradio.live'")
        print("\nOpening localhost instead...")
        return "http://localhost:7860"
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    url = find_gradio_url()
    if url:
        print(f"\n🌐 Opening: {url}")
        webbrowser.open(url)
    else:
        print("\n⚠️  Could not find public URL")
        print("The app may still be starting, or public URL sharing is not enabled")
        print("\nOpening localhost...")
        webbrowser.open('http://localhost:7860')

