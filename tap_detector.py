import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, HTML
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import time
from collections import deque
from PIL import Image
import concurrent.futures
from ultralytics import RTDETR
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tracking configuration
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_FRAMES_MISSING = 10
DEFAULT_HISTORY_LENGTH = 30

# Detection configuration
DEFAULT_CONF_THRESHOLD = 0.7
DEFAULT_FRAMES_PER_CHECK = 3

# Visual configuration
PERSON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128)
]
COLOR_NAMES = [
    "Red", "Green", "Blue", 
    "Yellow", "Magenta", "Cyan",
    "Maroon", "DarkGreen", "Navy"
]

# ============================================================================
# EVENT LOGGING WITH REAL-TIME BROADCASTING
# ============================================================================

class EventLogger:
    """Logger for tracking person detection and tap events during video processing"""
    
    def __init__(self, video_filename: str, broadcast_callback: Optional[Callable] = None):
        # Create logs directory if it doesn't exist
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Store broadcast callback for real-time updates
        self.broadcast_callback = broadcast_callback
        
        # Store person color mapping for real-time updates
        self.person_colors_rgb = {}  # {track_id: (r, g, b)}
        
        # Create event log filename based on video filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_base = Path(video_filename).stem
        
        # Create dedicated experiment folder
        self.experiment_folder = Path("experiments") / f"{video_base}_{timestamp}"
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for organization
        self.frames_folder = self.experiment_folder / "annotated_frames"
        self.frames_folder.mkdir(exist_ok=True)
        
        self.log_file = self.experiment_folder / f"events_{video_base}_{timestamp}.log"
        self.frame_counter = 0
        self.check_counter = 0
        
        # Buffer log entries to reduce I/O latency
        self.log_buffer = []
        self.log_buffer_size = 10  # Flush after 10 entries
        
        # Initialize the log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"VTA TRACKING - EVENT LOG\n")
            f.write(f"Video: {video_filename}\n")
            f.write(f"Processing Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        print(f"📝 Event log created: {self.log_file}")
    
    def _flush_log_buffer(self):
        """Flush buffered log entries to disk"""
        if self.log_buffer:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
    
    def _write_log(self, message: str):
        """Write log entry with buffering to reduce I/O"""
        self.log_buffer.append(message)
        if len(self.log_buffer) >= self.log_buffer_size:
            self._flush_log_buffer()
    
    def _broadcast_event(self, event_data: dict):
        """Broadcast event to connected clients via callback (non-blocking)"""
        if self.broadcast_callback:
            try:
                # Call in a non-blocking way to avoid latency
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule as a task without waiting
                        asyncio.create_task(self.broadcast_callback(event_data))
                    else:
                        # Fallback for sync context
                        asyncio.run(self.broadcast_callback(event_data))
                except RuntimeError:
                    # No event loop, skip broadcast to avoid blocking
                    pass
            except Exception as e:
                # Silently skip errors to avoid blocking processing
                pass
    
    def log_new_person(self, track_id: int, color_name: str, frame_number: int, is_initial: bool = False, color_rgb: Tuple[int, int, int] = None):
        """Log when a new person is detected"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Store color RGB for this person
        if color_rgb:
            self.person_colors_rgb[track_id] = color_rgb
        
        if is_initial:
            message = f"[{timestamp}] 🆕 INITIAL DETECTION - Person ID {track_id} ({color_name} box) detected at frame {frame_number}\n"
            event_type = "initial_detection"
        else:
            message = f"[{timestamp}] 🆕 NEW PERSON DETECTED - Person ID {track_id} ({color_name} box) entered at frame {frame_number}\n"
            event_type = "new_person"
        
        # Use buffered write to reduce I/O latency
        self._write_log(message)
        
        # Broadcast real-time update with complete person data
        self._broadcast_event({
            "type": "event",
            "event_type": event_type,
            "track_id": track_id,
            "color": color_name,
            "frame": frame_number,
            "timestamp": timestamp,
            "message": message.strip(),
            "experiment_folder": self.experiment_folder.name,  # Add experiment folder name
            # Additional data for real-time UI updates
            "person_data": {
                "track_id": track_id,
                "color": color_name,
                "color_rgb": list(color_rgb) if color_rgb else [128, 128, 128],
                "tapped": False,
                "first_seen_frame": frame_number,
                "total_frames_tracked": 0
            }
        })
    
    def log_tap_event(self, track_id: int, color_name: str, frame_number: int):
        """Log when a person taps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        message = f"[{timestamp}] ✅ TAP DETECTED - Person ID {track_id} ({color_name} box) tapped at frame {frame_number}\n"
        
        # Use buffered write to reduce I/O latency
        self._write_log(message)
        
        # Broadcast real-time update with tap information
        self._broadcast_event({
            "type": "event",
            "event_type": "tap_detected",
            "track_id": track_id,
            "color": color_name,
            "frame": frame_number,
            "timestamp": timestamp,
            "message": message.strip(),
            # Additional data for real-time UI updates
            "person_update": {
                "track_id": track_id,
                "tapped": True,
                "tap_frame": frame_number
            }
        })
    
    def log_summary(self, tracked_people: Dict, total_frames: int):
        """Log final summary at end of processing"""
        # Flush any buffered entries first
        self._flush_log_buffer()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tapped_count = sum(1 for p in tracked_people.values() if p.has_tapped)
        not_tapped_count = len(tracked_people) - tapped_count
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PROCESSING COMPLETED: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(f"Total People Tracked: {len(tracked_people)}\n")
            f.write(f"People Who Tapped: {tapped_count}\n")
            f.write(f"People Who Did Not Tap: {not_tapped_count}\n")
            f.write(f"\nDETAILED BREAKDOWN:\n")
            f.write(f"{'-'*80}\n")
            
            for track_id, person in sorted(tracked_people.items()):
                status = f"TAPPED at frame {person.tap_frame}" if person.has_tapped else "NO TAP"
                f.write(f"  Person ID {track_id} ({person.color_name} box): {status}\n")
            
            f.write(f"{'-'*80}\n")
        
        print(f"   📝 Summary logged to: {self.log_file}")
    
    def log_vlm_call(self, check_number: int, frame_numbers: List[int], people_colors: Dict[int, str], prompt: str):
        """Log VLM call details (optimized - skip full prompt)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Use buffered write for speed
        message = f"\n{'='*80}\n"
        message += f"[{timestamp}] 🔍 VLM CHECK #{check_number}\n"
        message += f"{'='*80}\n"
        message += f"Frames analyzed: {', '.join(map(str, frame_numbers))}\n"
        message += f"People being checked: {', '.join([f'{color} (ID {pid})' for pid, color in people_colors.items()])}\n\n"
        self._write_log(message)
    
    def log_vlm_response(self, check_number: int, response: str, parsed_results: Dict[int, Dict]):
        """Log VLM response and parsed results (optimized)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Use buffered write for speed
        message = f"[{timestamp}] 📥 VLM RESPONSE #{check_number}:\n"
        message += f"RESULTS:\n"
        for person_id, result in parsed_results.items():
            status = "TAPPED" if result['is_tapping'] else "NOT TAPPED"
            message += f"  Person ID {person_id}: {status}\n"
        message += "\n"
        self._write_log(message)
    
    def save_person_bbox_crop(self, frame_rgb: np.ndarray, bbox: np.ndarray, track_id: int, color: Tuple[int, int, int], color_name: str, frame_number: int):
        """Save full frame with colored bounding box for a person"""
        # Always print to console for visibility
        print(f"🖼️  Saving initial frame for Person {track_id} ({color_name}) at frame {frame_number}")
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Create a copy of the full frame
            frame_with_bbox = frame_rgb.copy()
            
            # Draw the bounding box on the full frame
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), color, 4)
            
            # Add color label at the top of bounding box
            label = f"{color_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(label, font, 0.8, 2)
            cv2.rectangle(frame_with_bbox,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        color, -1)
            cv2.putText(frame_with_bbox, label, 
                      (x1 + 5, y1 - 5),
                      font, 0.8, (255, 255, 255), 2)
            
            # Convert to BGR and save
            frame_bgr = cv2.cvtColor(frame_with_bbox, cv2.COLOR_RGB2BGR)
            frame_filename = f"person_{track_id}_{color_name}_frame_{str(frame_number).zfill(6)}.png"
            frame_path = self.frames_folder / frame_filename
            
            # Ensure folder exists (critical!)
            self.frames_folder.mkdir(parents=True, exist_ok=True)
            
            # Save the image with error checking
            success = cv2.imwrite(str(frame_path), frame_bgr)
            
            if success:
                print(f"   ✅ Successfully saved: {frame_filename}")
                # Verify file exists
                if not frame_path.exists():
                    print(f"   ⚠️  WARNING: File was 'saved' but doesn't exist at {frame_path}")
            else:
                print(f"   ❌ cv2.imwrite returned False for: {frame_filename}")
                print(f"   Frame shape: {frame_rgb.shape}")
                print(f"   Bbox: {bbox}")
                
        except Exception as e:
            print(f"   ❌ Exception saving frame for Person {track_id}: {str(e)}")
            print(f"   Frame shape: {frame_rgb.shape if frame_rgb is not None else 'None'}")
            print(f"   Bbox: {bbox}")
            import traceback
            traceback.print_exc()
    
    def save_annotated_frames_and_consolidate(self, annotated_frames: List[np.ndarray], 
                                               frame_numbers: List[int], 
                                               people_checked: Dict[int, str]) -> str:
        """Create a consolidated image - skip individual frame saves for speed"""
        self.check_counter += 1
        
        # Create ordinal suffix for call number
        def get_ordinal_suffix(n):
            if 10 <= n % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return suffix
        
        ordinal_suffix = get_ordinal_suffix(self.check_counter)
        call_name = f"{self.check_counter}{ordinal_suffix}_vlmcall.png"
        
        # Create consolidated image using matplotlib (skip individual frames for speed)
        if len(annotated_frames) > 0:
            # Use non-interactive backend for speed
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, len(annotated_frames), figsize=(6 * len(annotated_frames), 6))
            
            # Handle single frame case (axes won't be an array)
            if len(annotated_frames) == 1:
                axes = [axes]
            
            # Plot each frame
            for idx, (ax, frame, frame_num) in enumerate(zip(axes, annotated_frames, frame_numbers)):
                ax.imshow(frame)
                ax.set_title(f"Frame {frame_num}", fontsize=14, fontweight='bold')
                ax.axis('off')
            
            # Add overall title with metadata
            people_text = ", ".join([f"{color}" for color in people_checked.values()])
            fig.suptitle(f"VLM Check #{self.check_counter} | People: {people_text}", 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Quick layout (skip tight_layout for speed)
            plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)
            
            # Save consolidated image with lower DPI for speed
            consolidated_filename = self.frames_folder / call_name
            plt.savefig(str(consolidated_filename), dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return str(consolidated_filename)
        
        return ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_color_for_person(person_id: int) -> Tuple[Tuple[int, int, int], str]:
    """Get unique color and name for a person ID"""
    idx = person_id % len(PERSON_COLORS)
    return PERSON_COLORS[idx], COLOR_NAMES[idx]

# ============================================================================
# SmolVLM TAP DETECTOR
# ============================================================================

class SmolVLMTapDetector:
    """Tap detection using SmolVLM with color-coded bounding boxes"""

    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct"):
        print("📦 Loading SmolVLM...")
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        print("✅ SmolVLM loaded")

    def detect_tap_multi_frame(self, frames: List[np.ndarray], person_colors: Dict[int, str], 
                              event_logger: Optional['EventLogger'] = None, check_number: int = 0, 
                              frame_numbers: List[int] = None) -> Dict[int, Dict]:
        """Detect if multiple people tapped across multiple frames using color-coded boxes"""
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)

        person_ids = list(person_colors.keys())
        content = []
        
        for img in pil_frames:
            content.append({"type": "image"})

        # ===================================================================
        # PROMPT VARIABLES - Available for customization:
        # ===================================================================
        # {num_frames}     - Number of frames being analyzed (e.g., 1, 2, or 3)
        # {colors_list}    - Comma-separated list of colors present (e.g., "RED, BLUE, GREEN")
        # ===================================================================
        
        num_frames = len(pil_frames)  # Variable: Number of frames (1-3)
        colors_order = list(person_colors.values())  # e.g. ["GREEN","YELLOW"] or ["YELLOW"]
        colors_list = ", ".join(colors_order)  # Variable: Comma-separated colors (e.g., "RED, BLUE")

        prompt_text = f"""
        You are analyzing {num_frames} security camera frames.

        People are identified ONLY by the colored bounding box around them.
        VALID COLORS (use ONLY these): {colors_list}

        TASK:
        Determine whether each color-box person TAPPED the fare payment reader.

        DEFINITION OF “TAPPED” (be strict):
        Mark a person as TAPPED only if, in at least one frame, you clearly see ALL of:
        1) The person’s hand/arm extends toward the fare reader
        2) The hand/card/phone touches the reader OR is within ~2 inches of it
        3) The motion is a clear payment gesture (not just standing near, walking by, or reaching elsewhere)

        NOT A TAP (always NOT TAPPED):
        - Simply standing closest to the reader
        - Walking past the reader
        - Hand not extended toward the reader
        - Reader is not clearly visible / interaction is occluded
        - Unclear distance or uncertain contact

        ANTI-BIAS RULES (mandatory):
        - Do NOT guess based on who is closest to the reader.
        - Do NOT infer a tap from posture alone.
        - If you cannot clearly verify contact/within-2-inches + a payment gesture, output NOT TAPPED.

        OUTPUT FORMAT (exactly two lines, nothing else):
        TAPPED: [comma-separated list of COLOR OR NONE]
        NOT TAPPED: [comma-separated list of COLOR OR NONE]

        Strictly follow this CONSTRAINTS:
        - Each color must appear in exactly ONE category (TAPPED or NOT TAPPED)
        - If TAPPED is NONE, then NOT TAPPED must list ALL colors: {colors_list}
        - If NOT TAPPED is NONE, then TAPPED must list ALL colors: {colors_list}

        Now analyze the frames and respond in the required format.
        """.strip()

        content.append({"type": "text", "text": prompt_text})
        
        # Log VLM call if logger is provided
        if event_logger:
            event_logger.log_vlm_call(check_number, frame_numbers or list(range(len(frames))), person_colors, prompt_text)

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=pil_frames, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
            )

        generated_ids = output[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        print(f"\n{'='*70}")
        print(f"📥 VLM RESPONSE: {response}")
        print(f"{'='*70}\n")

        # Parse response
        results = {}
        response_upper = response.upper().strip()

        for person_id in person_colors.keys():
            results[person_id] = {'is_tapping': False, 'response': response}

        valid_colors = set(c.upper() for c in person_colors.values())
        tapped_colors = []

        lines = response_upper.split('\n')
        for line in lines:
            if 'TAPPED:' in line and 'NOT TAPPED:' not in line:
                content = line.split('TAPPED:')[-1].strip()
                content = content.replace('[', '').replace(']', '').strip()
                
                if 'NONE' not in content:
                    parsed = [c.strip() for c in content.replace(',', ' ').split() if c.strip()]
                    tapped_colors = [c for c in parsed if c in valid_colors]

        for person_id, color_name in person_colors.items():
            color_upper = color_name.upper()
            if color_upper in tapped_colors:
                results[person_id]['is_tapping'] = True
                results[person_id]['response'] = f"{color_name}: TAPPED"
            else:
                results[person_id]['response'] = f"{color_name}: NOT TAPPED"
        
        # Log VLM response if logger is provided
        if event_logger:
            event_logger.log_vlm_response(check_number, response, results)

        return results

# ============================================================================
# TRACKED PERSON DATA
# ============================================================================

@dataclass
class TrackedPerson:
    track_id: int
    color: Tuple[int, int, int]
    color_name: str
    has_tapped: bool = False
    tap_frame: Optional[int] = None
    frame_count: int = 0
    last_check_frame: int = -1
    bbox: Optional[List[float]] = None
    first_seen_frame: int = 0

class MultiPersonTapTracker:
    def __init__(self, rtdetr_model="rtdetr-x.pt", conf_threshold=0.7, frames_per_check=3):
        print(f"📦 Loading RT-DETR model: {rtdetr_model}...")
        self.rtdetr = RTDETR(rtdetr_model)
        self.conf_threshold = conf_threshold
        self.tap_detector = SmolVLMTapDetector()
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.frames_per_check = frames_per_check
        self.event_logger = None
        self.frame_buffer = []
        self.people_in_buffer = set()
        self.tapped_people: Dict[int, bool] = {}
        self.experiment_folder = None
        print("✅ RT-DETR loaded")

    def initialize_tracking(self, video_path, initial_frame=0):
        """Initialize with first tracking call"""
        print(f"\n{'='*70}")
        print(f"🎬 Initializing Multi-Person Tap Detection")
        print(f"   Initial frame: {initial_frame}")
        print(f"{'='*70}\n")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ Could not read frame")
            return 0

        results = self.rtdetr.track(
            frame,
            classes=[0],
            conf=self.conf_threshold,
            persist=True,
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, track_id in zip(boxes, track_ids):
                color, color_name = get_color_for_person(track_id)
                self.tracked_people[track_id] = TrackedPerson(
                    track_id=track_id,
                    color=color,
                    color_name=color_name,
                    bbox=bbox.tolist(),
                    first_seen_frame=initial_frame
                )
                self.tapped_people[track_id] = False
                # Skip verbose print for speed
                # print(f"   ✓ Initialized Person {track_id} with {color_name} box")
                
                if self.event_logger:
                    # Save cropped bbox image for this person
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.event_logger.save_person_bbox_crop(frame_rgb, bbox, track_id, color, color_name, initial_frame)
                    # Pass color RGB for frontend display
                    self.event_logger.log_new_person(track_id, color_name, initial_frame, is_initial=True, color_rgb=color)

        return len(self.tracked_people)

    def track_all_people(self, video_path, check_interval=30, initial_frame=0, broadcast_callback=None):
        """Track using color-coded bounding boxes with real-time event broadcasting"""
        # Initialize event logger with broadcast callback
        self.event_logger = EventLogger(video_path, broadcast_callback=broadcast_callback)
        self.experiment_folder = self.event_logger.experiment_folder
        
        num_people = self.initialize_tracking(video_path, initial_frame)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

        video_detections = {}
        print(f"\n🔄 Processing video with real-time event broadcasting...\n")

        frame_idx = initial_frame
        vlm_check_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if len(self.frame_buffer) > 0:
                    self.check_buffered_frames(frame_idx)
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.rtdetr.track(
                frame,
                classes=[0],
                conf=self.conf_threshold,
                persist=True,
                verbose=False
            )

            current_detections = {}

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if track_id not in self.tracked_people:
                        color, color_name = get_color_for_person(track_id)
                        self.tracked_people[track_id] = TrackedPerson(
                            track_id=track_id,
                            color=color,
                            color_name=color_name,
                            first_seen_frame=frame_idx
                        )
                        self.tapped_people[track_id] = False
                        # Skip verbose print for speed during processing
                        # print(f"   New person {track_id} ({color_name} box) at frame {frame_idx}")
                        
                        if self.event_logger:
                            # Save cropped bbox image for this person
                            self.event_logger.save_person_bbox_crop(frame_rgb, box, track_id, color, color_name, frame_idx)
                            self.event_logger.log_new_person(track_id, color_name, frame_idx, is_initial=False, color_rgb=color)

                    current_detections[track_id] = {
                        'bbox': box,
                        'conf': conf,
                        'track_id': track_id
                    }

                    tracked_person = self.tracked_people[track_id]
                    tracked_person.frame_count += 1
                    tracked_person.bbox = box.tolist()

            video_detections[frame_idx] = current_detections

            # Buffer management
            base_frame = vlm_check_counter * check_interval + initial_frame
            target_frames = [base_frame + 10, base_frame + 20, base_frame + 30]

            if frame_idx in target_frames:
                if len(current_detections) > 0:
                    # Only copy frame if we need it - deep copy is expensive
                    self.frame_buffer.append((frame_idx, frame_rgb.copy(), dict(current_detections)))
                    self.people_in_buffer.update(current_detections.keys())

            if frame_idx == base_frame + 30 and len(self.frame_buffer) > 0:
                self.check_buffered_frames(frame_idx)
                vlm_check_counter += 1

            frame_idx += 1

        cap.release()
        self.save_results_to_json()

        if self.event_logger:
            self.event_logger.log_summary(self.tracked_people, frame_idx)

        return video_detections, self.tracked_people

    def check_buffered_frames(self, frame_idx):
        """Check buffered frames and broadcast tap events in real-time"""
        # Skip only if buffer is empty, otherwise process whatever frames we have (1-3)
        if len(self.frame_buffer) == 0:
            # Skip verbose print for speed
            # print(f"   ℹ️  No frames in buffer, skipping VLM check")
            self.people_in_buffer.clear()
            return

        people_to_check = sorted([tid for tid in self.people_in_buffer
                                  if tid in self.tracked_people and not self.tapped_people.get(tid, False)])

        if not people_to_check:
            # Skip verbose print for speed
            # print(f"   ℹ️  No people to check (all already tapped or no detections)")
            self.frame_buffer.clear()
            self.people_in_buffer.clear()
            return

        person_colors = {tid: self.tracked_people[tid].color_name for tid in people_to_check}

        annotated_frames = []
        for frame_num, frame_rgb, people_dict in self.frame_buffer:
            frame_annotated = frame_rgb.copy()

            for track_id in people_to_check:
                if track_id in people_dict:
                    bbox = people_dict[track_id]['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    color = self.tracked_people[track_id].color
                    color_name = self.tracked_people[track_id].color_name

                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 4)

                    label = f"{color_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.8, 2)
                    cv2.rectangle(frame_annotated,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width + 10, y1),
                                color, -1)
                    cv2.putText(frame_annotated, label, (x1 + 5, y1 - 5),
                              font, 0.8, (255, 255, 255), 2)

            annotated_frames.append(frame_annotated)

        # Save annotated frames and create consolidated image
        frame_numbers = [frame_num for frame_num, _, _ in self.frame_buffer]
        if self.event_logger:
            # Note: Individual person frames are already saved when first detected
            # via save_person_bbox_crop() in initialize_tracking() and track_all_people()
            # We only need to save the consolidated VLM check image here
            consolidated_path = self.event_logger.save_annotated_frames_and_consolidate(
                annotated_frames, frame_numbers, person_colors
            )

        results = self.tap_detector.detect_tap_multi_frame(
            annotated_frames, 
            person_colors,
            event_logger=self.event_logger,
            check_number=self.event_logger.check_counter if self.event_logger else 0,
            frame_numbers=frame_numbers
        )

        for track_id, result in results.items():
            if track_id in self.tracked_people:
                tracked_person = self.tracked_people[track_id]
                if result['is_tapping'] and not self.tapped_people.get(track_id, False):
                    tracked_person.has_tapped = True
                    tracked_person.tap_frame = frame_idx
                    self.tapped_people[track_id] = True
                    print(f"\n   ✅ {tracked_person.color_name} box (ID {track_id}) TAPPED at frame {frame_idx}!")
                    
                    if self.event_logger:
                        self.event_logger.log_tap_event(track_id, tracked_person.color_name, frame_idx)

        self.frame_buffer.clear()
        self.people_in_buffer.clear()

    def save_results_to_json(self, output_file="tap_detection_results.json"):
        """Save tracking results to JSON file"""
        import json

        results = {
            "summary": {
                "total_people": len(self.tracked_people),
                "people_tapped": sum(1 for p in self.tracked_people.values() if p.has_tapped),
                "people_not_tapped": sum(1 for p in self.tracked_people.values() if not p.has_tapped)
            },
            "people": []
        }

        for track_id, person in sorted(self.tracked_people.items()):
            person_data = {
                "track_id": int(track_id),
                "color": person.color_name,
                "color_rgb": [int(c) for c in person.color],
                "tapped": bool(person.has_tapped),
                "tap_frame": int(person.tap_frame) if person.has_tapped and person.tap_frame is not None else None,
                "total_frames_tracked": int(person.frame_count)
            }
            results["people"].append(person_data)

        # Save to experiment folder if available
        if self.experiment_folder:
            output_file = self.experiment_folder / "results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

# Visualization function (simplified version)
def visualize_multi_person_tracking(video_path, video_detections, tracked_people,
                                   output_path='multi_person_tap_output.mp4',
                                   initial_frame=0):
    """Create video with color-coded multi-person tracking"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)
    frame_idx = initial_frame
    max_frame = max(video_detections.keys()) if video_detections else initial_frame

    print("🎬 Creating output video...")

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in video_detections:
            for person_id, detection in video_detections[frame_idx].items():
                if person_id in tracked_people:
                    person = tracked_people[person_id]
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    color = person.color

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    if person.has_tapped and frame_idx >= person.tap_frame:
                        status = "PAID ✓"
                        bg_color = (0, 200, 0)
                    else:
                        status = "Waiting..."
                        bg_color = (50, 50, 50)

                    label = f"{person.color_name}: {status}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width + 10, y1),
                                bg_color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                              font, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Video saved: {output_path}")