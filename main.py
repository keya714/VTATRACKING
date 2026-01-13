from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Dict, List
import cv2
import numpy as np

# Configuration constants
DEFAULT_CONF_THRESHOLD = 0.7
DEFAULT_FRAMES_PER_CHECK = 3
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8501
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Setup directories
LOG_DIR = Path("logs")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
EXPERIMENTS_DIR = Path("experiments")

for directory in [LOG_DIR, UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'tap_detection_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tap Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Global tracker - models loaded at startup
tracker = None

@app.on_event("startup")
async def startup_event():
    """Initialize models when server starts"""
    global tracker
    try:
        logger.info("🔄 Loading RT-DETR and SmolVLM models...")
        from tap_detector import MultiPersonTapTracker
        tracker = MultiPersonTapTracker(
            rtdetr_model="rtdetr-x.pt",
            conf_threshold=DEFAULT_CONF_THRESHOLD,
            frames_per_check=DEFAULT_FRAMES_PER_CHECK
        )
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}")

class ProcessingStatus:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.message = ""
        self.current_file = ""
        
    def update(self, status: str, progress: int, message: str):
        self.status = status
        self.progress = progress
        self.message = message
        logger.info(f"Status Update: {status} - {progress}% - {message}")

processing_status = ProcessingStatus()

async def broadcast_status(data: dict):
    """Broadcast status to all connected WebSocket clients"""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            disconnected.append(connection)
    
    for connection in disconnected:
        if connection in active_connections:
            active_connections.remove(connection)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the main interface"""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/static/index.html">
        </head>
        <body>
            <p>Redirecting to <a href="/static/index.html">Tap Detection System</a>...</p>
        </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(active_connections)}")
    
    try:
        await websocket.send_json({
            "status": processing_status.status,
            "progress": processing_status.progress,
            "message": processing_status.message
        })
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                await websocket.send_json({"type": "ping", "data": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        if not file.filename.endswith(VIDEO_EXTENSIONS):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid file type. Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully: {file_path}")
        
        file_size = os.path.getsize(file_path)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "upload",
            "filename": filename,
            "original_name": file.filename,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "status": "success"
        }
        
        with open(LOG_DIR / "upload_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "file_size_mb": log_entry["file_size_mb"],
            "message": "Video uploaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/process/{filename}")
async def process_video(
    filename: str,
    conf_threshold: float = 0.7,
    frames_per_check: int = 3,
    check_interval: int = 30,
    initial_frame: int = 0
):
    """Process uploaded video for tap detection with real-time events"""
    try:
        # Check if models are loaded
        if tracker is None:
            logger.error("Models not loaded")
            raise HTTPException(status_code=503, detail="Models are still loading. Please wait and try again.")
        
        video_path = UPLOAD_DIR / filename
        
        if not video_path.exists():
            logger.error(f"Video file not found: {filename}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        logger.info(f"Starting processing for: {filename}")
        processing_status.update("processing", 0, "Initializing tracker...")
        processing_status.current_file = filename
        
        await broadcast_status({
            "status": "processing",
            "progress": 5,
            "message": "Preparing tracker..."
        })
        
        from tap_detector import visualize_multi_person_tracking
        
        # Update tracker parameters for this request
        tracker.conf_threshold = conf_threshold
        tracker.frames_per_check = frames_per_check
        
        await broadcast_status({
            "status": "processing",
            "progress": 10,
            "message": "Starting tracking..."
        })
        
        logger.info("Starting multi-person tracking with real-time events...")
        video_detections, tracked_people = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tracker.track_all_people(
                video_path=str(video_path),
                check_interval=check_interval,
                initial_frame=initial_frame,
                broadcast_callback=lambda data: asyncio.create_task(broadcast_status(data))
            )
        )
        
        await broadcast_status({
            "status": "processing",
            "progress": 70,
            "message": "Tracking complete. Generating output video..."
        })
        
        output_filename = f"processed_{filename}"
        output_path = OUTPUT_DIR / output_filename
        
        logger.info("Creating visualization video...")
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: visualize_multi_person_tracking(
                video_path=str(video_path),
                video_detections=video_detections,
                tracked_people=tracked_people,
                output_path=str(output_path),
                initial_frame=initial_frame
            )
        )
        
        # Copy processed video to experiment folder
        if tracker.experiment_folder:
            experiment_video_path = tracker.experiment_folder / f"processed_video{output_path.suffix}"
            shutil.copy(str(output_path), str(experiment_video_path))
            logger.info(f"Copied processed video to experiment folder: {experiment_video_path}")
            
            # Create a summary file in experiment folder
            summary_path = tracker.experiment_folder / "experiment_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("VTA TRACKING EXPERIMENT SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Experiment Folder: {tracker.experiment_folder.name}\n")
                f.write(f"Input Video: {filename}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("PROCESSING PARAMETERS:\n")
                f.write(f"  - Confidence Threshold: {conf_threshold}\n")
                f.write(f"  - Frames Per Check: {frames_per_check}\n")
                f.write(f"  - Check Interval: {check_interval}\n")
                f.write(f"  - Initial Frame: {initial_frame}\n\n")
                f.write("FOLDER CONTENTS:\n")
                f.write(f"  - annotated_frames/     : Individual annotated frames sent to VLM\n")
                f.write(f"  - consolidated_images/  : Consolidated images of each VLM check\n")
                f.write(f"  - events_*.log          : Detailed event log (new person, tap detection)\n")
                f.write(f"  - results.json          : JSON results with all tracked people\n")
                f.write(f"  - processed_video.*     : Final output video with tracking\n")
                f.write(f"  - experiment_summary.txt: This file\n\n")
                f.write("="*80 + "\n")
            
            logger.info(f"All experiment files organized in: {tracker.experiment_folder}")
        
        await broadcast_status({
            "status": "processing",
            "progress": 90,
            "message": "Saving results..."
        })
        
        results = {
            "summary": {
                "total_people": len(tracked_people),
                "people_tapped": sum(1 for p in tracked_people.values() if p.has_tapped),
                "people_not_tapped": sum(1 for p in tracked_people.values() if not p.has_tapped)
            },
            "people": []
        }
        
        for track_id, person in sorted(tracked_people.items()):
            person_data = {
                "track_id": int(track_id),
                "color": person.color_name,
                "color_rgb": [int(c) for c in person.color],
                "tapped": bool(person.has_tapped),
                "tap_frame": int(person.tap_frame) if person.has_tapped and person.tap_frame else None,
                "total_frames_tracked": int(person.frame_count),
                "first_seen_frame": int(person.first_seen_frame)
            }
            results["people"].append(person_data)
        
        results_filename = f"results_{filename.rsplit('.', 1)[0]}.json"
        results_path = OUTPUT_DIR / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "process",
            "input_file": filename,
            "output_video": output_filename,
            "results_file": results_filename,
            "summary": results["summary"],
            "processing_params": {
                "conf_threshold": conf_threshold,
                "frames_per_check": frames_per_check,
                "check_interval": check_interval,
                "initial_frame": initial_frame
            },
            "status": "success"
        }
        
        with open(LOG_DIR / "processing_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"Processing complete for {filename}")
        
        processing_status.update("complete", 100, "Processing complete")
        
        await broadcast_status({
            "status": "complete",
            "progress": 100,
            "message": "Processing complete!"
        })
        
        return JSONResponse({
            "status": "success",
            "output_video": output_filename,
            "results_file": results_filename,
            "experiment_folder": str(tracker.experiment_folder.name) if tracker.experiment_folder else None,
            "results": results,
            "message": "Video processed successfully"
        })
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        
        await broadcast_status({
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}"
        })
        
        processing_status.update("error", 0, str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/download/video/{filename}")
async def download_video(filename: str):
    """Download processed video"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"Downloading video: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )

@app.get("/api/download/results/{filename}")
async def download_results(filename: str):
    """Download results JSON"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"Downloading results: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/json"
    )

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return JSONResponse({
        "status": processing_status.status,
        "progress": processing_status.progress,
        "message": processing_status.message,
        "current_file": processing_status.current_file
    })

@app.get("/api/logs")
async def get_logs(log_type: str = "events", limit: int = 100):
    """Get recent event log entries (new person detection and tap detection)"""
    try:
        # Find the most recent events_*.log file in experiments folders
        event_logs = []
        
        # Search through all experiment folders
        if EXPERIMENTS_DIR.exists():
            for experiment_folder in EXPERIMENTS_DIR.iterdir():
                if experiment_folder.is_dir():
                    for log_file in experiment_folder.glob("events_*.log"):
                        event_logs.append(log_file)
        
        # Sort by modification time, most recent first
        event_logs = sorted(event_logs, key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not event_logs:
            return JSONResponse({"logs": [], "log_file": None})
        
        log_file = event_logs[0]  # Most recent event log
        logger.info(f"Reading event log: {log_file}")
        
        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header lines (first 5 lines with ====)
            content_lines = [line for line in lines if not line.startswith('=') and line.strip()]
            
            # Parse only event lines (those with timestamps and emoji indicators)
            for line in content_lines[-limit:]:
                if any(indicator in line for indicator in ['🆕', '✅']):
                    # Extract timestamp and message
                    if '[' in line and ']' in line:
                        timestamp_end = line.index(']')
                        timestamp = line[1:timestamp_end]
                        message = line[timestamp_end + 2:].strip()
                        
                        # Determine event type
                        if 'INITIAL DETECTION' in message:
                            event_type = 'initial_detection'
                        elif 'NEW PERSON DETECTED' in message:
                            event_type = 'new_person'
                        elif 'TAP DETECTED' in message:
                            event_type = 'tap_detected'
                        else:
                            event_type = 'unknown'
                        
                        logs.append({
                            "timestamp": timestamp,
                            "message": message,
                            "event_type": event_type,
                            "raw_line": line.strip()
                        })
        
        return JSONResponse({
            "logs": logs,
            "log_file": log_file.name,
            "experiment_folder": log_file.parent.name,
            "total_events": len(logs)
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch event logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/event-log-files")
async def get_event_log_files():
    """Get list of all event log files from experiments folders"""
    try:
        event_logs = []
        
        # Search through all experiment folders
        if EXPERIMENTS_DIR.exists():
            for experiment_folder in EXPERIMENTS_DIR.iterdir():
                if experiment_folder.is_dir():
                    for log_file in experiment_folder.glob("events_*.log"):
                        event_logs.append(log_file)
        
        # Sort by modification time, most recent first
        event_logs = sorted(event_logs, key=lambda p: p.stat().st_mtime, reverse=True)
        
        log_files = []
        for log_file in event_logs:
            stat = log_file.stat()
            log_files.append({
                "filename": log_file.name,
                "experiment_folder": log_file.parent.name,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(log_file)
            })
        
        return JSONResponse({
            "log_files": log_files,
            "total": len(log_files)
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch event log files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_websockets": len(active_connections)
    })

@app.get("/api/frame/{experiment_folder}/{frame_filename}")
async def get_frame_image(experiment_folder: str, frame_filename: str):
    """Serve a specific frame image from an experiment folder"""
    try:
        frame_path = EXPERIMENTS_DIR / experiment_folder / "annotated_frames" / frame_filename
        
        logger.info(f"📷 Frame requested: {frame_filename} from {experiment_folder}")
        logger.info(f"   Full path: {frame_path}")
        logger.info(f"   Exists: {frame_path.exists()}")
        
        if not frame_path.exists():
            # List what files DO exist in that folder
            folder_path = EXPERIMENTS_DIR / experiment_folder / "annotated_frames"
            if folder_path.exists():
                existing_files = list(folder_path.glob("*.png"))
                logger.warning(f"   Frame not found. Existing files: {[f.name for f in existing_files[:5]]}")
            else:
                logger.error(f"   Folder doesn't exist: {folder_path}")
            raise HTTPException(status_code=404, detail="Frame image not found")
        
        return FileResponse(
            path=frame_path,
            media_type="image/png",
            filename=frame_filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve frame image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    if not (STATIC_DIR / "index.html").exists():
        logger.warning("⚠️  index.html not found in static/ directory!")
        logger.info("Please save the frontend HTML to static/index.html")
    
    logger.info("="*70)
    logger.info("🚀 Starting Tap Detection API Server with Real-time Events")
    logger.info("="*70)
    logger.info(f"📂 Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")
    logger.info(f"📂 Experiments directory: {EXPERIMENTS_DIR.absolute()}")
    logger.info(f"📂 Static files: {STATIC_DIR.absolute()}")
    logger.info("="*70)
    logger.info(f"🌐 Server will start at: http://localhost:{SERVER_PORT}")
    logger.info(f"🌐 Web Interface: http://localhost:{SERVER_PORT}/static/index.html")
    logger.info(f"📚 API Documentation: http://localhost:{SERVER_PORT}/docs")
    logger.info("="*70)
    
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")