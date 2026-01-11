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
        
        # Create event log filename based on video filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_base = Path(video_filename).stem
        self.log_file = self.log_dir / f"events_{video_base}_{timestamp}.log"
        
        # Initialize the log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"VTA TRACKING - EVENT LOG\n")
            f.write(f"Video: {video_filename}\n")
            f.write(f"Processing Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
        
        print(f"📝 Event log created: {self.log_file}")
    
    def _broadcast_event(self, event_data: dict):
        """Broadcast event to connected clients via callback"""
        if self.broadcast_callback:
            try:
                # Call the async broadcast function from sync context
                # The callback should handle the async execution
                self.broadcast_callback(event_data)
            except Exception as e:
                print(f"   ⚠️  Failed to broadcast event: {e}")
    
    def log_new_person(self, track_id: int, color_name: str, frame_number: int, is_initial: bool = False):
        """Log when a new person is detected"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        if is_initial:
            message = f"[{timestamp}] 🆕 INITIAL DETECTION - Person ID {track_id} ({color_name} box) detected at frame {frame_number}\n"
            event_type = "initial_detection"
        else:
            message = f"[{timestamp}] 🆕 NEW PERSON DETECTED - Person ID {track_id} ({color_name} box) entered at frame {frame_number}\n"
            event_type = "new_person"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
        
        print(f"   📝 Logged: {message.strip()}")
        
        # Broadcast real-time update
        self._broadcast_event({
            "type": "event",
            "event_type": event_type,
            "track_id": track_id,
            "color": color_name,
            "frame": frame_number,
            "timestamp": timestamp,
            "message": message.strip()
        })
    
    def log_tap_event(self, track_id: int, color_name: str, frame_number: int):
        """Log when a person taps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        message = f"[{timestamp}] ✅ TAP DETECTED - Person ID {track_id} ({color_name} box) tapped at frame {frame_number}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
        
        print(f"   📝 Logged: {message.strip()}")
        
        # Broadcast real-time update
        self._broadcast_event({
            "type": "event",
            "event_type": "tap_detected",
            "track_id": track_id,
            "color": color_name,
            "frame": frame_number,
            "timestamp": timestamp,
            "message": message.strip()
        })
    
    def log_summary(self, tracked_people: Dict, total_frames: int):
        """Log final summary at end of processing"""
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

# ============================================================================
# COLOR MAPPING FOR PERSON IDENTIFICATION
# ============================================================================

PERSON_COLORS = [
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta/Pink
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 128),    # Purple
]

COLOR_NAMES = [
    "RED",
    "BLUE",
    "GREEN",
    "YELLOW",
    "PINK",
    "CYAN",
    "ORANGE",
    "PURPLE"
]

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

    def detect_tap_multi_frame(self, frames: List[np.ndarray], person_colors: Dict[int, str]) -> Dict[int, Dict]:
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

        colors_order = list(person_colors.values())
        colors_list = ", ".join(colors_order)

        prompt_text = f"""
You are analyzing {len(pil_frames)} security camera frames.

People are identified ONLY by the colored bounding box around them.
VALID COLORS (use ONLY these): {colors_list}

TASK:
Determine whether each color-box person TAPPED the fare payment reader.

DEFINITION OF "TAPPED" (be strict):
Mark a person as TAPPED only if, in at least one frame, you clearly see ALL of:
1) The person's hand/arm extends toward the fare reader
2) The hand/card/phone touches the reader OR is within ~2 inches of it
3) The motion is a clear payment gesture (not just standing near, walking by, or reaching elsewhere)

NOT A TAP (always NOT TAPPED):
- Simply standing closest to the reader
- Walking past the reader
- Hand not extended toward the reader
- Reader is not clearly visible / interaction is occluded
- Unclear distance or uncertain contact

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
                    bbox=bbox.tolist()
                )
                self.tapped_people[track_id] = False
                print(f"   ✓ Initialized Person {track_id} with {color_name} box")
                
                if self.event_logger:
                    self.event_logger.log_new_person(track_id, color_name, initial_frame, is_initial=True)

        return len(self.tracked_people)

    def track_all_people(self, video_path, check_interval=30, initial_frame=0, broadcast_callback=None):
        """Track using color-coded bounding boxes with real-time event broadcasting"""
        # Initialize event logger with broadcast callback
        self.event_logger = EventLogger(video_path, broadcast_callback=broadcast_callback)
        
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
                            color_name=color_name
                        )
                        self.tapped_people[track_id] = False
                        print(f"   New person {track_id} ({color_name} box) at frame {frame_idx}")
                        
                        if self.event_logger:
                            self.event_logger.log_new_person(track_id, color_name, frame_idx, is_initial=False)

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
                    self.frame_buffer.append((frame_idx, frame_rgb.copy(), current_detections.copy()))
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
        if len(self.frame_buffer) < self.frames_per_check:
            return

        people_to_check = sorted([tid for tid in self.people_in_buffer
                                  if tid in self.tracked_people and not self.tapped_people.get(tid, False)])

        if not people_to_check:
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

        results = self.tap_detector.detect_tap_multi_frame(annotated_frames, person_colors)

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