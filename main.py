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

# Configure detailed logging
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
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
    
    # Remove disconnected clients
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
    logger.info(f"WebSocket connection established. Total connections: {len(active_connections)}")
    
    try:
        # Send current status immediately on connect
        await websocket.send_json({
            "status": processing_status.status,
            "progress": processing_status.progress,
            "message": processing_status.message
        })
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back to keep alive
                await websocket.send_json({"type": "ping", "data": "pong"})
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully: {file_path}")
        
        # Log upload details
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
    """Process uploaded video for tap detection"""
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
        
        # Import here to avoid loading models on startup
        from tap_detector import MultiPersonTapTracker, visualize_multi_person_tracking
        
        # Create tracker
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
        
        # Process video
        logger.info("Starting multi-person tracking...")
        video_detections, tracked_people = tracker.track_all_people(
            video_path=str(video_path),
            check_interval=check_interval,
            initial_frame=initial_frame
        )
        
        await broadcast_status({
            "status": "processing",
            "progress": 70,
            "message": "Tracking complete. Generating output video..."
        })
        
        # Create output video
        output_filename = f"processed_{filename}"
        output_path = OUTPUT_DIR / output_filename
        
        logger.info("Creating visualization video...")
        visualize_multi_person_tracking(
            video_path=str(video_path),
            video_detections=video_detections,
            tracked_people=tracked_people,
            output_path=str(output_path),
            initial_frame=initial_frame
        )
        
        await broadcast_status({
            "status": "processing",
            "progress": 90,
            "message": "Saving results..."
        })
        
        # Prepare results
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
        
        # Save results JSON
        results_filename = f"results_{filename.rsplit('.', 1)[0]}.json"
        results_path = OUTPUT_DIR / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log processing completion
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
async def get_logs(log_type: str = "processing", limit: int = 50):
    """Get recent log entries"""
    try:
        log_file = LOG_DIR / f"{log_type}_log.jsonl"
        
        if not log_file.exists():
            return JSONResponse({"logs": []})
        
        logs = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return JSONResponse({"logs": logs})
        
    except Exception as e:
        logger.error(f"Failed to fetch logs: {str(e)}")
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
    # Create static directory and save index.html if it doesn't exist
    if not (STATIC_DIR / "index.html").exists():
        logger.warning("⚠️  index.html not found in static/ directory!")
        logger.info("Please save the frontend HTML to static/index.html")
    
    logger.info("="*70)
    logger.info("🚀 Starting Tap Detection API Server")
    logger.info("="*70)
    logger.info(f"📂 Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")
    logger.info(f"📂 Static files: {STATIC_DIR.absolute()}")
    logger.info("="*70)
    logger.info("🌐 Server will start at: http://localhost:8000")
    logger.info("🌐 Web Interface: http://localhost:8000/static/index.html")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")