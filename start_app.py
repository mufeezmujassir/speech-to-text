#!/usr/bin/env python3
"""
Simple Speech-to-Text App Launcher
Usage: python run.py
"""

import subprocess
import time
import webbrowser
import os
import sys
import threading
from pathlib import Path

def run_backend():
    """Run the backend server"""
    backend_dir = Path("backend")
    if backend_dir.exists():
        os.chdir(backend_dir)
        subprocess.run([sys.executable, "app.py"])

def run_frontend():
    """Run the frontend server"""
    frontend_dir = Path("frontend")
    if frontend_dir.exists():
        os.chdir(frontend_dir)
        subprocess.run([sys.executable, "-m", "http.server", "8000"])

def main():
    print("Starting Speech-to-Text Application...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    time.sleep(5)
    
    # Start frontend in a separate thread  
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait for frontend to start
    time.sleep(3)
    
    # Open browser
    print("Opening browser...")
    webbrowser.open("http://localhost:8000/frontend/")
    
    print("\nApplication started!")
    print("Backend: http://localhost:5000")
    print("Frontend: http://localhost:8000/frontend/") 
    print("\nPress Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
        sys.exit(0)

if __name__ == "__main__":
    main()