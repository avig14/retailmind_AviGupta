"""
run.py — Entry point for RetailMind AI Agent.

Usage:
    python run.py

This launches the Streamlit app on the default port (8501).
"""

import subprocess
import sys
import os


def main():
    # Resolve the path to app.py relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")

    print("Starting RetailMind AI Agent -- StyleCraft Intelligence Dashboard")
    print(f"App: {app_path}")
    print("Open http://localhost:8501 in your browser")
    print("(Streamlit may open it automatically)\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "false"],
        cwd=script_dir,
    )


if __name__ == "__main__":
    main()
