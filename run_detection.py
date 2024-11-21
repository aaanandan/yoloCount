import os
import sys
import subprocess
import webbrowser
import time
import platform

def check_environment():
    """Check if virtual environment is activated"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def setup_environment():
    """Setup virtual environment and install dependencies"""
    if not check_environment():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "yolo_env"])
        
        # Get path to Python and pip in virtual environment
        if platform.system() == "Windows":
            python_executable = os.path.join("yolo_env", "Scripts", "python.exe")
            pip_executable = os.path.join("yolo_env", "Scripts", "pip.exe")
        else:
            python_executable = os.path.join("yolo_env", "bin", "python")
            pip_executable = os.path.join("yolo_env", "bin", "pip")
        
        if os.path.exists(python_executable):
            print("Installing dependencies in virtual environment...")
            # First upgrade pip
            subprocess.run([pip_executable, "install", "--upgrade", "pip"])
            
            required_packages = [
                'torch',
                'opencv-python',
                'flask',
                'yt-dlp',
                'pandas',
                'seaborn',
                'matplotlib',
                'tqdm',
                'PyYAML',
                'requests',
                'psutil',
                'numpy',
                'scipy'
            ]
            
            for package in required_packages:
                print(f"Installing {package}...")
                subprocess.run([pip_executable, "install", package])
            
            print("Restarting with virtual environment...")
            # Add PYTHONPATH environment variable
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_dir = os.path.join(current_dir, "yolov7")
            os.environ["PYTHONPATH"] = yolo_dir
            os.execv(python_executable, [python_executable] + sys.argv)
        else:
            print("Failed to find virtual environment Python. Please activate it manually:")
            print(f"On Windows: {os.path.join('yolo_env', 'Scripts', 'activate')}")
            print(f"On Unix: source {os.path.join('yolo_env', 'bin', 'activate')}")
            sys.exit(1)

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import torch
        import cv2
        import flask
        import pafy
        import youtube_dl
        print("All required packages are installed.")
        
        # Print versions
        print("\nPackage versions:")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        print(f"Flask: {flask.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def launch_browser():
    """Launch web browser to view the stream"""
    print("Waiting for server to start...")
    time.sleep(5)  # Wait for Flask server to start
    url = "http://localhost:5000/video"
    webbrowser.open(url)
    print(f"Opened browser at {url}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_detection.py <youtube_url>")
        sys.exit(1)
        
    youtube_url = sys.argv[1]
    
    # Setup and check environment
    setup_environment()
    check_dependencies()
    
    # Add yolov7 directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(current_dir, "yolov7")
    sys.path.append(yolo_dir)
    
    # Run YOLO detection
    print("\nStarting YOLO detection...")
    subprocess.Popen([sys.executable, "run_yolo.py", youtube_url], 
                    env=dict(os.environ, PYTHONPATH=yolo_dir))
    
    # Launch browser
    launch_browser()

if __name__ == "__main__":
    main() 