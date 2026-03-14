"""
run_app.py
Convenience script to launch the Streamlit dashboard.
"""

import os
import sys
import subprocess


def run():
    """Launch streamlit app."""
    # Ensure current directory is in path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    app_path = os.path.join(current_dir, "app", "streamlit_app.py")
    
    print("🚀 Starting AI Quant Research Sandbox...")
    print(f"📍 App path: {app_path}")
    
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        print("💡 Hint: Make sure streamlit is installed: pip install streamlit")


if __name__ == "__main__":
    run()
