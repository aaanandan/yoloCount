import os
import cv2
import sys
import yaml
import logging
import torch
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
from flask import Flask, Response, stream_with_context, jsonify
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import non_max_suppression
from yolov7.utils.torch_utils import select_device
from queue import Queue
from threading import Thread
import time
from youtube_dl import YoutubeDL
import pafy
import psutil
from datetime import datetime
import numpy as np

# Setup logging
def setup_logging(config):
    handlers = []
    if config['logging'].get('file'):
        handlers.append(logging.FileHandler(config['logging']['file']))
    if config['logging'].get('console', True):
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=getattr(logging, config['logging'].get('level', 'INFO')),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = 'config.yaml'
def load_config():
    default_config = {
        'model': {
            'weights_path': 'yolov7/yolov7.pt',
            'img_size': 640,
            'conf_thres': 0.5,
            'iou_thres': 0.5
        },
        'video': {
            'output_dir': 'yolov7/runs/detect/exp',
            'save_stream': True,
            'batch_size': 32,
            'output_format': 'mp4',
            'save_frames': False,
            'compression': 95
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8081
        },
        'stream': {
            'buffer_size': 30,
            'frame_skip': 0,
            'processing_delay': 0.0,
            'youtube_quality': 'best',
            'target_fps': 30,
            'max_resolution': 1080,
            'rtsp_buffer_size': 1024,
            'reconnect_attempts': 3,
            'timeout': 30
        },
        'logging': {
            'level': 'INFO',
            'file': 'yolo_detection.log',
            'console': True
        }
    }
    
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = default_config
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f)
    
    return config

# Initialize global variables
config = load_config()
logger = setup_logging(config)
app = Flask(__name__)
device = None
model = None
frame_queue = Queue(maxsize=config['stream']['buffer_size'])
processing_complete = False
frame_count = 0
start_time = time.time()
total_people_count = 0
people_tracking = {}
DETECTION_LINE_Y = None  # Will be set based on frame height
DETECTION_THRESHOLD = 30  # pixels threshold for counting

class ResourceManager:
    def __init__(self):
        self.resources = []
        self.start_time = time.time()
    
    def add(self, resource):
        self.resources.append(resource)
    
    def cleanup(self):
        for resource in self.resources:
            try:
                resource.release()
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")
    
    def monitor_resources(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()/1024**2
            gpu_memory_cached = torch.cuda.memory_reserved()/1024**2
            logger.info(f"GPU Memory: Used={gpu_memory_used:.2f}MB, Cached={gpu_memory_cached:.2f}MB")
            
        logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")
        
        if cpu_percent > 90 or memory_percent > 90:
            logger.warning("System resources are running high!")

resource_manager = ResourceManager()

def initialize_model():
    global device, model
    try:
        weights_path = config['model']['weights_path']
        if not os.path.exists(weights_path):
            logger.error(f"Model weights not found at {weights_path}")
            return False
            
        device = select_device("0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        weights = torch.load(weights_path, map_location=device)
        model = weights['model'].float()
        
        if device.type != 'cpu':
            model = model.half()
            
        model.to(device).eval()
        
        # Model warmup
        dummy_input = torch.zeros(1, 3, config['model']['img_size'], 
                                config['model']['img_size']).to(device)
        if device.type != 'cpu':
            dummy_input = dummy_input.half()
        model(dummy_input)
        
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

def download_youtube_video(url):
    try:
        logger.info("Processing YouTube video...")
        import yt_dlp
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            logger.info(f"Video title: {info.get('title', 'Unknown')}")
            logger.info(f"Duration: {info.get('duration', 'Unknown')} seconds")
            logger.info(f"Resolution: {info.get('resolution', 'Unknown')}")
            
            return video_url
            
    except Exception as e:
        logger.error(f"Error processing YouTube video: {e}")
        return None

def validate_url(url):
    try:
        if url.isdigit():
            return {'type': 'webcam', 'valid': True, 'source': int(url)}
            
        if url.startswith('file:///'):
            path = url.replace('file:///', '')
            if os.path.exists(path):
                return {'type': 'local', 'valid': True, 'source': path}
                
        if 'youtube.com' in url or 'youtu.be' in url:
            return {'type': 'youtube', 'valid': True, 'source': None}
            
        if url.lower().startswith('rtsp://'):
            return {'type': 'rtsp', 'valid': True, 'source': url}
            
        if url.lower().startswith(('http://', 'https://')):
            if any(ext in url.lower() for ext in ['.mp4', '.m3u8', '.mpd', 'mjpeg']):
                return {'type': 'stream', 'valid': True, 'source': url}
            
            response = requests.head(url)
            content_type = response.headers.get('content-type', '')
            if any(t in content_type for t in ['video', 'stream', 'mpegurl']):
                return {'type': 'stream', 'valid': True, 'source': url}
                
        return {'type': 'unknown', 'valid': False, 'source': None}
    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return {'type': 'unknown', 'valid': False, 'source': None}

def process_frame(frame):
    max_height = config['stream'].get('max_resolution', 1080)
    if frame.shape[0] > max_height:
        scale = max_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale)
        frame = cv2.resize(frame, (new_width, max_height))
    return frame

def detect(frame):
    global device, model
    if model is None:
        if not initialize_model():
            return {"pred": [], "count": 0}

    try:
        frame = process_frame(frame)
        img = cv2.resize(frame, (config['model']['img_size'],) * 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()
        img /= 255.0
        img = img.unsqueeze(0)

        torch.cuda.empty_cache()

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(
                pred, 
                config['model']['conf_thres'],
                config['model']['iou_thres']
            )[0]

        results = []
        count = 0
        if pred is not None and len(pred):
            scale_x = frame.shape[1] / config['model']['img_size']
            scale_y = frame.shape[0] / config['model']['img_size']
            
            for det in pred:
                if int(det[5]) == 0:  # person class
                    x1, y1, x2, y2 = det[:4]
                    x1, x2 = int(x1.item() * scale_x), int(x2.item() * scale_x)
                    y1, y2 = int(y1.item() * scale_y), int(y2.item() * scale_y)
                    results.append([x1, y1, x2, y2])
                    count += 1

        return {"pred": results, "count": count}
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return {"pred": [], "count": 0}

def update_people_count(frame, detections):
    global total_people_count, DETECTION_LINE_Y, people_tracking
    
    if DETECTION_LINE_Y is None:
        DETECTION_LINE_Y = int(frame.shape[0] * 0.5)  # Set line at middle of frame
    
    current_frame_ids = set()
    
    # Draw counting line
    cv2.line(frame, (0, DETECTION_LINE_Y), (frame.shape[1], DETECTION_LINE_Y), (255, 0, 0), 2)
    
    # Process each detection
    for det in detections["pred"]:
        x1, y1, x2, y2 = map(int, det)
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Create a unique ID based on position (can be improved with actual tracking)
        person_id = f"{x1}_{x2}"
        current_frame_ids.add(person_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Initialize tracking for new person
        if person_id not in people_tracking:
            people_tracking[person_id] = {
                "first_seen": time.time(),
                "last_position": person_center,
                "counted": False,
                "last_seen": time.time(),
                "crossed_line": False,
                "direction": None
            }
        
        current_y = person_center[1]
        last_y = people_tracking[person_id]["last_position"][1]
        
        # Determine crossing direction
        if not people_tracking[person_id]["crossed_line"]:
            # Person is approaching the line from above
            if last_y < DETECTION_LINE_Y and current_y >= DETECTION_LINE_Y:
                people_tracking[person_id]["direction"] = "down"
                if not people_tracking[person_id]["counted"]:
                    total_people_count += 1
                    people_tracking[person_id]["counted"] = True
                    people_tracking[person_id]["crossed_line"] = True
                    
            # Person is approaching the line from below
            elif last_y > DETECTION_LINE_Y and current_y <= DETECTION_LINE_Y:
                people_tracking[person_id]["direction"] = "up"
                if not people_tracking[person_id]["counted"]:
                    total_people_count += 1
                    people_tracking[person_id]["counted"] = True
                    people_tracking[person_id]["crossed_line"] = True
        
        # Update last known position
        people_tracking[person_id]["last_position"] = person_center
        people_tracking[person_id]["last_seen"] = time.time()
        
        # Draw ID and direction on frame
        direction = people_tracking[person_id]["direction"] or "unknown"
        cv2.putText(frame, f"ID: {person_id[:4]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Clean up old tracking entries (remove after 2 seconds of not being seen)
    current_time = time.time()
    people_tracking = {
        k: v for k, v in people_tracking.items()
        if current_time - v["last_seen"] < 2.0 or k in current_frame_ids
    }
    
    # Draw total count on frame
    cv2.putText(frame, f"Total Count: {total_people_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame
def process_video_stream(url):
    global processing_complete, frame_count, total_people_count
    processing_complete = False
    logger.info("Starting video processing...")

    url_info = validate_url(url)
    if not url_info['valid']:
        logger.error("Invalid URL provided")
        return False

    try:
        if url_info['type'] == 'youtube':
            source = download_youtube_video(url)
            if not source:
                return False
        else:
            source = url_info['source']

        # Create unique output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config['video']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = output_dir / f'processed_video_{timestamp}.mp4'
        stats_path = output_dir / f'statistics_{timestamp}.json'
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Error: Could not open video source")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = detect(frame)
            
            # Update people count and draw on frame (this now handles all drawing)
            frame = update_people_count(frame, results)
            
            # Save frame to video
            out.write(frame)
            
            # Add to streaming queue
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)

        # Save statistics
        stats = {
            "total_people_count": total_people_count,
            "processed_frames": frame_count,
            "processing_date": timestamp,
            "video_source": url,
            "fps": fps,
            "duration_seconds": frame_count / fps if fps > 0 else 0
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
        logger.info(f"Processed video saved to: {video_path}")
        logger.info(f"Statistics saved to: {stats_path}")
        logger.info(f"Total people counted: {total_people_count}")
        
        return True

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        return False
    finally:
        processing_complete = True
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

@app.route("/video")
def video_feed():
    def generate():
        while not processing_complete or not frame_queue.empty():
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, jpeg = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, config['video']['compression']])
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(stream_with_context(generate()),
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "processing": not processing_complete,
        "frames_processed": frame_count,
        "device": str(device),
        "model_loaded": model is not None,
        "elapsed_time": time.time() - start_time,
        "fps": frame_count / (time.time() - start_time) if frame_count > 0 else 0
    })

@app.route("/count")
def get_count():
    return jsonify({
        "total_people_count": total_people_count,
        "timestamp": datetime.now().isoformat()
    })

def start_servers():
    process_thread = Thread(target=process_video_stream, args=(video_url,))
    process_thread.start()
    
    logger.info(f"Stream available at http://{config['server']['host']}:{config['server']['port']}/video")
    logger.info(f"Status available at http://{config['server']['host']}:{config['server']['port']}/status")
    
    app.run(
        host=config['server']['host'],
        port=config['server']['port'],
        threaded=True
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python run_yolo.py <video_url>")
        sys.exit(1)

    video_url = sys.argv[1]
    try:
        start_servers()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        resource_manager.cleanup()
    except Exception as e:
        logger.error(f"Application error: {e}")
        resource_manager.cleanup()
        sys.exit(1)
