# Real-Time People Counter with YOLOv7

A robust people detection and counting system that uses YOLOv7 for real-time detection, with video streaming capabilities and comprehensive statistics tracking. The system counts people as they cross a detection line and provides real-time statistics through a web interface.

## Key Features

- Real-time people detection and counting using YOLOv7
- Support for multiple video sources:
  - YouTube videos
  - RTSP streams
  - Local video files
  - Webcam feeds
- Live video streaming through web interface
- Real-time statistics and count tracking
- JSON-based statistics export
- Resource monitoring (CPU, GPU, Memory usage)
- Configurable settings via YAML
- Detection line crossing counter
- Individual person numbering

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Git
- 4GB RAM minimum (8GB+ recommended)
- Webcam (optional, for live camera feed)

## Installation

1. Clone the repository:

```

2. Create and activate virtual environment:
```

3. Install dependencies:

````

## Configuration

The system is configured through `config.yaml`. Key settings include:

```yaml
model:
  weights_path: "yolov7/yolov7.pt"
  img_size: 640
  conf_thres: 0.5

video:
  output_dir: "yolov7/runs/detect/exp"
  save_stream: true
  compression: 95

server:
  host: "0.0.0.0"
  port: 5000

location:
  name: "Main Entrance"
  timezone: "Your Timezone"
````

## Usage

1. Start the detection system:

```bash
python run_yolo.py <video_source>
```

Example sources:

- YouTube: `https://youtube.com/watch?v=example`
- RTSP: `rtsp://camera.example.com/stream`
- Webcam: `0` (for default camera)
- Local file: `path/to/video.mp4`

2. Access the web interface:

- Video stream: `http://localhost:5000/video`
- Statistics: `http://localhost:5000/stats`
- Count data: `http://localhost:5000/count`

## Web Interface Endpoints

| Endpoint     | Description                                    |
| ------------ | ---------------------------------------------- |
| `/video`     | Live video stream with detection visualization |
| `/stats`     | Current statistics in JSON format              |
| `/stats/all` | Historical statistics                          |
| `/count`     | Current people count                           |
| `/status`    | System status and performance metrics          |

## Output Files

The system generates:

1. Processed Videos

   - Location: `yolov7/runs/detect/exp/`
   - Format: MP4
   - Includes: Detection line, person numbers, count

2. Statistics Files
   - Format: JSON
   - Contains:
     - Total people count
     - Processing timestamp
     - Frame statistics
     - Location data
     - Duration metrics

## Features in Detail

### Detection System

- YOLOv7 for accurate person detection
- Real-time processing
- GPU acceleration (when available)
- Configurable confidence thresholds

### Counting Mechanism

- Detection line crossing counter
- Individual person numbering
- Double-counting prevention
- Real-time count updates

### Video Processing

- Multiple input source support
- Automatic YouTube video processing
- RTSP stream handling
- Frame rate optimization
- Resolution adjustment

### Statistics Tracking

- Real-time count updates
- JSON statistics export
- Timestamp tracking
- Location-based data
- Performance metrics

## Troubleshooting

### Common Issues

1. CUDA/GPU Issues:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Video Source Problems:

- Verify URL/path accessibility
- Check network connection
- Confirm camera permissions

3. Performance Issues:

- Reduce image size in config
- Lower FPS settings
- Monitor resource usage

### Resource Management

The system includes automatic resource monitoring:

- CPU usage tracking
- Memory utilization
- GPU memory (if available)
- Automatic cleanup

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv7 by WongKinYiu
- OpenCV community
- Flask framework

```

This README provides comprehensive documentation covering:
- Installation and setup
- Configuration options
- Usage instructions
- Feature details
- Troubleshooting guides
- Contributing guidelines