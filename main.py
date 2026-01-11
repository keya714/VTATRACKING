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

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

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

def sync_broadcast(data: dict):
    """Synchronous wrapper for broadcasting from sync context"""
    try:
        # Create event loop if needed and run the async broadcast
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task
            asyncio.create_task(broadcast_status(data))
        else:
            # If loop is not running, run it
            loop.run_until_complete(broadcast_status(data))
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broadcast_status(data))
        loop.close()

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
        
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
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
        video_path = UPLOAD_DIR / filename
        
        if not video_path.exists():
            logger.error(f"Video file not found: {filename}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        logger.info(f"Starting processing for: {filename}")
        processing_status.update("processing", 0, "Initializing tracker...")
        processing_status.current_file = filename
        
        await broadcast_status({
            "status": "processing",
            "progress": 0,
            "message": "Initializing RT-DETR and SmolVLM models..."
        })
        
        from tap_detector import MultiPersonTapTracker, visualize_multi_person_tracking
        
        tracker = MultiPersonTapTracker(
            rtdetr_model="rtdetr-x.pt",
            conf_threshold=conf_threshold,
            frames_per_check=frames_per_check
        )
        
        await broadcast_status({
            "status": "processing",
            "progress": 10,
            "message": "Models loaded. Starting tracking..."
        })
        
        # Use sync_broadcast as the callback for real-time events
        logger.info("Starting multi-person tracking with real-time events...")
        video_detections, tracked_people = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tracker.track_all_people(
                video_path=str(video_path),
                check_interval=check_interval,
                initial_frame=initial_frame,
                broadcast_callback=sync_broadcast  # Pass sync wrapper
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
                "total_frames_tracked": int(person.frame_count)
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
        
        await broadcast_status({
            "status": "complete",
            "progress": 100,
            "message": "Processing complete!"
        })
        
        processing_status.update("complete", 100, "Processing complete")
        
        return JSONResponse({
            "status": "success",
            "output_video": output_filename,
            "results_file": results_filename,
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
        # Find the most recent events_*.log file
        event_logs = sorted(LOG_DIR.glob("events_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
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
            "total_events": len(logs)
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch event logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/event-log-files")
async def get_event_log_files():
    """Get list of all event log files"""
    try:
        event_logs = sorted(LOG_DIR.glob("events_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        log_files = []
        for log_file in event_logs:
            stat = log_file.stat()
            log_files.append({
                "filename": log_file.name,
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
    logger.info(f"📂 Static files: {STATIC_DIR.absolute()}")
    logger.info("="*70)
    logger.info("🌐 Server will start at: http://localhost:8501")
    logger.info("🌐 Web Interface: http://localhost:8501/static/index.html")
    logger.info("📚 API Documentation: http://localhost:8501/docs")
    logger.info("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")