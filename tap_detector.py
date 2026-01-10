# ============================================================================
# VISUALIZATION
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, HTML
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
from collections import deque
from PIL import Image
import concurrent.futures
from ultralytics import RTDETR

# ============================================================================
# COLOR MAPPING FOR PERSON IDENTIFICATION
# ============================================================================

# Using only highly distinguishable primary colors that VLMs easily recognize
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
    "PINK",      # More recognizable than "MAGENTA"
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
        """
        Detect if multiple people tapped across multiple frames using color-coded boxes

        Args:
            frames: List of image crops (2-3 frames showing same people)
            person_colors: Dict of {person_id: color_name}

        Returns:
            Dict of {person_id: {'is_tapping': bool, 'response': str}}
        """
        # Convert all frames to PIL
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            else:
                pil_frames.append(frame)

        # Create color list description
        color_list = ", ".join([f"{color} box" for color in person_colors.values()])
        person_ids = list(person_colors.keys())

        content = []
        for img in pil_frames:
            content.append({"type": "image"})




        # Build a strict, no-placeholder, persona-based prompt
        colors_order = list(person_colors.values())  # e.g. ["GREEN","YELLOW"] or ["YELLOW"]
        color_list_block = "\n".join([f"- {c} box" for c in colors_order])
        colors_list = ", ".join(colors_order)

        prompt_text=f"""
        You are analyzing {len(pil_frames)} security camera frames.

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

        # prompt_text = f"""You are analyzing {len(pil_frames)} security camera frames. Each person has a colored box.

        # COLORS IN THESE FRAMES: {colors_list}

        # TASK: Determine if each colored box person TAPPED the fare payment reader.

        # WHAT IS A TAP? (Be very strict)
        # A person TAPPED only if you see ALL of these:
        # 1. Their hand/arm extends toward a payment device/reader
        # 2. Their hand/card/phone makes contact OR gets within 2 inches of the reader
        # 3. There is a clear payment gesture (not just walking by or standing)

        # WHAT IS NOT A TAP?
        # - Just walking past the reader
        # - Standing near it without extending hand
        # - Hand in pocket or at their side
        # - Cannot see the reader clearly
        # - Person is too far away
        # - Uncertain or blurry motion

        # CRITICAL RULES:
        # - ONLY use colors from this list: {colors_list}
        # - Do NOT mention colors not in the list above
        # - When in doubt, answer NOT TAPPED
        # - Each color must appear in exactly ONE category

        # OUTPUT FORMAT (required):
        # TAPPED: [colors who tapped, or NONE]
        # NOT TAPPED: [colors who did not tap, or NONE]

        # Example 1 -  if GREEN tapped, BLUE did not:
        # TAPPED: GREEN
        # NOT TAPPED: BLUE

        # Example 2 - Nobody tapped:
        # TAPPED: NONE
        # NOT TAPPED: {colors_list}

        # Analyze the {len(pil_frames)} frames above and respond now:"""

        # prompt_text = f"""Watch these {len(pil_frames)} frames. People are marked with colored boxes: {colors_list}

        # Did each person TAP the payment reader with their hand/card/phone?

        # Respond in this exact format:
        # TAPPED: [list colors who tapped, or write NONE]
        # NOT TAPPED: [list colors who did not tap, or write NONE]

        # Example if GREEN tapped but BLUE did not:
        # TAPPED: GREEN
        # NOT TAPPED: BLUE

        # Example if nobody tapped:
        # TAPPED: NONE
        # NOT TAPPED: {colors_list}

        # Now analyze and respond:"""

        # prompt_text = f"""
        # You are a meticulous transit-fare compliance analyst reviewing {len(pil_frames)} frames from a kiosk security camera.
        # People are identified ONLY by the COLORED bounding box around them.

        # ONLY these colored boxes exist in the images:
        # {color_list_block}

        # Per color:
        # - If tapped, output exactly: "<COLOR> box: YES"
        # - If not tapped, output: "<COLOR> box: NO "

        # Tap definition (YES):
        # Answer YES for a given bounding box color only if you clearly see a payment interaction near the fare reader:
        # - hand/arm moves toward the reader AND
        # - hand/card/phone is held close to the reader (pause/hold) OR a clear tap/contact occurs.

        # Tap definition (NO):
        # Answer NO for a given bounding box color only if the above conditions for YES are not satisfied.

        # Conservative policy:
        # If the reader is not visible, occluded, or you are uncertain: treat as NOT tapped.

        # OUTPUT REQUIREMENTS (mandatory):
        # - Output EXACTLY {len(colors_order)} line(s), and NOTHING else.
        # - Each line MUST start with the exact prefix "<COLOR> box:" (including the word 'box').
        # - Use ONLY the listed colors. Do NOT output any other colors.

        # Prohibited:
        # - Do NOT output only a color name (e.g., "YELLOW").
        # - Do NOT output instructions, templates, or placeholder text.
        # - Do NOT use angle brackets or placeholder wording.
        # - If you cannot comply perfectly, output exactly: ERROR

        # """.strip()

        # # Add color-based format examples
        # for person_id, color_name in person_colors.items():
        #     prompt_text += f"{color_name} box: YES or NO\n"

        # prompt_text += (
        #     f"\nExample response:\n"
        #     f"{list(person_colors.values())[0]} box: YES\n"
        #     f"{list(person_colors.values())[1] if len(person_colors) > 1 else list(person_colors.values())[0]} box: NO\n\n"
        #     f"Now analyze the frames and respond:"
        # )

        content.append({"type": "text", "text": prompt_text})

        # Print debug info
        print(f"\n{'='*70}")
        print(f"🔍 SENDING TO VLM:")
        print(f"   Number of frames: {len(pil_frames)}")
        print(f"   Person colors: {person_colors}")
        print(f"   Prompt preview: {prompt_text}")
        print(f"{'='*70}\n")

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=prompt,
            images=pil_frames,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
            )

        # Decode only generated tokens
        generated_ids = output[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Print VLM response
        print(f"\n{'='*70}")
        print(f"📥 VLM RESPONSE:")
        print(f"   Raw response: {response}")
        print(f"{'='*70}\n")

        # # Parse response for each person based on color
        # results = {}
        # lines = response.strip().split('\n')

        # for person_id, color_name in person_colors.items():
        #     # Try to find the response for this person's color
        #     person_response = "NO"  # Default
        #     is_tapping = False

        #     for line in lines:
        #         line_upper = line.upper()
        #         color_upper = color_name.upper()

        #         # Look for pattern like "RED box: YES" or "RED BOX: YES"
        #         if color_upper in line_upper and "BOX" in line_upper:
        #             person_response = line.strip()
        #             # Check if YES is in this specific line
        #             if 'YES' in line_upper:
        #                 is_tapping = True
        #             break

        #     results[person_id] = {
        #         'is_tapping': is_tapping,
        #         'response': person_response
        #     }
        # Parse response for categorized format

        results = {}
        response_upper = response.upper().strip()

        # Initialize all as not tapped
        for person_id in person_colors.keys():
            results[person_id] = {
                'is_tapping': False,
                'response': response
            }

        # Get valid colors from person_colors (prevent hallucination)
        valid_colors = set(c.upper() for c in person_colors.values())

        # Look for TAPPED: and NOT TAPPED: lines
        tapped_colors = []
        not_tapped_colors = []

        lines = response_upper.split('\n')
        for line in lines:
            if 'TAPPED:' in line and 'NOT TAPPED:' not in line:
                # Extract everything after "TAPPED:"
                content = line.split('TAPPED:')[-1].strip()

                # Remove square brackets if present
                content = content.replace('[', '').replace(']', '').strip()

                if 'NONE' not in content:
                    # Parse colors (could be comma-separated or space-separated)
                    parsed = [c.strip() for c in content.replace(',', ' ').split() if c.strip()]
                    # Filter out hallucinated colors
                    tapped_colors = [c for c in parsed if c in valid_colors]

            elif 'NOT TAPPED:' in line:
                content = line.split('NOT TAPPED:')[-1].strip()

                # Remove square brackets if present
                content = content.replace('[', '').replace(']', '').strip()

                if 'NONE' not in content:
                    parsed = [c.strip() for c in content.replace(',', ' ').split() if c.strip()]
                    not_tapped_colors = [c for c in parsed if c in valid_colors]

        # Mark people who tapped (only if their color appears in tapped_colors)
        for person_id, color_name in person_colors.items():
            color_upper = color_name.upper()
            if color_upper in tapped_colors:
                results[person_id]['is_tapping'] = True
                results[person_id]['response'] = f"{color_name}: TAPPED"
            else:
                results[person_id]['response'] = f"{color_name}: NOT TAPPED"

        # Debug output
        print(f"   Parsed - Tapped colors: {tapped_colors}")
        print(f"   Parsed - Not tapped colors: {not_tapped_colors}")
        print(f"   Valid colors in frame: {list(valid_colors)}")

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

        self.frame_buffer = []
        self.people_in_buffer = set()

        # Global map to track who has tapped (by color)
        self.tapped_people: Dict[int, bool] = {}  # {track_id: True/False}

        print("✅ RT-DETR loaded")

    def initialize_tracking(self, video_path, initial_frame=0):
        """Initialize with first tracking call - assign colors once and keep them"""
        print(f"\n{'='*70}")
        print(f"🎬 Initializing Multi-Person Tap Detection with Color-Coded Boxes")
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
                # Assign color ONCE per track_id
                color, color_name = get_color_for_person(track_id)
                self.tracked_people[track_id] = TrackedPerson(
                    track_id=track_id,
                    color=color,
                    color_name=color_name,
                    bbox=bbox.tolist()
                )
                self.tapped_people[track_id] = False  # Initialize as not tapped
                print(f"   ✓ Initialized Person {track_id} with {color_name} box (will keep this color throughout)")

        num_people = len(self.tracked_people)
        print(f"\n   Number of people initialized: {num_people}")
        return num_people

    def track_all_people(self, video_path, check_interval=30, initial_frame=0):
        """Track using color-coded bounding boxes with frames at 10, 20, 30 pattern"""
        num_people = self.initialize_tracking(video_path, initial_frame)

        # if num_people == 0:
        #     print("❌ No people detected in initial frame")
        #     return {}, {}

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

        video_detections = {}
        print(f"\n🔄 Processing video (VLM checks on frames 10,20,30 then 40,50,60, etc.)...\n")

        frame_idx = initial_frame
        vlm_check_counter = 0  # Track which VLM check we're on

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
                        # New person appeared mid-video - assign color once
                        color, color_name = get_color_for_person(track_id)
                        self.tracked_people[track_id] = TrackedPerson(
                            track_id=track_id,
                            color=color,
                            color_name=color_name
                        )
                        self.tapped_people[track_id] = False  # Initialize as not tapped
                        print(f"   New person {track_id} ({color_name} box) detected at frame {frame_idx} - color will remain consistent")

                    current_detections[track_id] = {
                        'bbox': box,
                        'conf': conf,
                        'track_id': track_id
                    }

                    tracked_person = self.tracked_people[track_id]
                    tracked_person.frame_count += 1
                    tracked_person.bbox = box.tolist()

            video_detections[frame_idx] = current_detections

            # Buffer management - collect frames 10, 20, 30 then 40, 50, 60, etc.
            # Pattern: base_frame = vlm_check_counter * 30 + initial_frame
            # Collect: base_frame + 10, base_frame + 20, base_frame + 30
            base_frame = vlm_check_counter * check_interval + initial_frame
            target_frames = [base_frame + 10, base_frame + 20, base_frame + 30]

            if frame_idx in target_frames:
                if len(current_detections) > 0:
                    self.frame_buffer.append((frame_idx, frame_rgb.copy(), current_detections.copy()))
                    self.people_in_buffer.update(current_detections.keys())
                    colors_in_frame = [self.tracked_people[tid].color_name for tid in current_detections.keys()]
                    print(f"   📸 Buffered frame {frame_idx} with colors: {colors_in_frame} (buffer size: {len(self.frame_buffer)}/3)")

            # Check when we have all 3 frames (after frame 30, 60, 90, etc.)
            if frame_idx == base_frame + 30 and len(self.frame_buffer) > 0:
                print(f"   🔍 Triggering VLM check with frames: {[f[0] for f in self.frame_buffer]}")
                self.check_buffered_frames(frame_idx)
                vlm_check_counter += 1

            if frame_idx % 30 == 0:
                tapped_count = sum(1 for p in self.tracked_people.values() if p.has_tapped)
                print(f"   Frame {frame_idx} | People tapped: {tapped_count}/{len(self.tracked_people)}")

            frame_idx += 1

        cap.release()

        # Save results to JSON
        self.save_results_to_json()

        print(f"\n📊 Results Summary:")
        for track_id, person in self.tracked_people.items():
            status = f"✅ TAPPED (frame {person.tap_frame})" if person.has_tapped else "❌ NO TAP"
            print(f"   {person.color_name} box (ID {track_id}): {status}")

        return video_detections, self.tracked_people

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
                "track_id": int(track_id),  # Convert to Python int
                "color": person.color_name,
                "color_rgb": [int(c) for c in person.color],  # Convert to Python ints
                "tapped": bool(person.has_tapped),  # Convert to Python bool
                "tap_frame": int(person.tap_frame) if person.has_tapped and person.tap_frame is not None else None,
                "total_frames_tracked": int(person.frame_count)  # Convert to Python int
            }
            results["people"].append(person_data)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    def check_buffered_frames(self, frame_idx):
        """Check buffered frames with color-coded annotations - skip people who already tapped"""
        if len(self.frame_buffer) < self.frames_per_check:
            return

        # ONLY check people who haven't tapped yet (using global tapped_people map)
        people_to_check = sorted([tid for tid in self.people_in_buffer
                                  if tid in self.tracked_people and not self.tapped_people.get(tid, False)])

        if not people_to_check:
            print(f"   ⏭️  Skipping VLM check - all people in buffer have already tapped")
            self.frame_buffer.clear()
            self.people_in_buffer.clear()
            return

        # Print frames being sent to VLM
        frame_numbers = [f[0] for f in self.frame_buffer]
        people_colors = [self.tracked_people[tid].color_name for tid in people_to_check]
        print(f"\n{'='*70}")
        print(f"📤 SENDING TO SmolVLM:")
        print(f"   Frame numbers: {frame_numbers}")
        print(f"   Number of frames: {len(self.frame_buffer)}")
        print(f"   People to check: {people_to_check} (colors: {people_colors})")
        print(f"   Already tapped (skipping): {[tid for tid in self.people_in_buffer if self.tapped_people.get(tid, False)]}")
        print(f"{'='*70}\n")

        # Create mapping of person_id to color name
        person_colors = {tid: self.tracked_people[tid].color_name for tid in people_to_check}

        annotated_frames = []
        for frame_num, frame_rgb, people_dict in self.frame_buffer:
            frame_annotated = frame_rgb.copy()

            for track_id in people_to_check:
                if track_id in people_dict:
                    bbox = people_dict[track_id]['bbox']
                    x1, y1, x2, y2 = map(int, bbox)

                    # Get color for this person
                    color = self.tracked_people[track_id].color
                    color_name = self.tracked_people[track_id].color_name

                    # Draw colored bounding box (thick)
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 4)

                    # Add color label
                    label = f"{color_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.8, 2)
                    cv2.rectangle(frame_annotated,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width + 10, y1),
                                color, -1)

                    # White text
                    cv2.putText(frame_annotated, label, (x1 + 5, y1 - 5),
                              font, 0.8, (255, 255, 255), 2)

            annotated_frames.append(frame_annotated)

        # Print summary before sending to VLM
        print(f"\n{'='*70}")
        print(f"🎨 ANNOTATED FRAMES READY:")
        print(f"   Frames: {frame_numbers}")
        print(f"   Colors in frames: {list(person_colors.values())}")
        print(f"   Sending to SmolVLM now...")
        print(f"{'='*70}\n")

        # Visualize frames being sent to VLM
        self._display_frames_to_vlm(annotated_frames, frame_numbers, person_colors)

        # Send to VLM with color mapping
        results = self.tap_detector.detect_tap_multi_frame(annotated_frames, person_colors)

        for track_id, result in results.items():
            if track_id in self.tracked_people:
                tracked_person = self.tracked_people[track_id]
                if result['is_tapping'] and not self.tapped_people.get(track_id, False):
                    tracked_person.has_tapped = True
                    tracked_person.tap_frame = frame_idx
                    self.tapped_people[track_id] = True  # Mark in global map
                    print(f"\n   ✅ {tracked_person.color_name} box (ID {track_id}) TAPPED at frame {frame_idx}!")
                    print(f"   🔒 {tracked_person.color_name} box will no longer be sent to VLM")

        self.frame_buffer.clear()
        self.people_in_buffer.clear()

    def _display_frames_to_vlm(self, frames, frame_numbers, person_colors):
        """Display the actual frames being sent to VLM using matplotlib"""
        num_frames = len(frames)

        fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 6))

        # Handle single frame case
        if num_frames == 1:
            axes = [axes]

        for idx, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
            axes[idx].imshow(frame)
            axes[idx].set_title(f'Frame {frame_num}', fontsize=14, fontweight='bold')
            axes[idx].axis('off')

        # Add overall title with color information
        colors_str = ", ".join([f"{color} box" for color in person_colors.values()])
        fig.suptitle(f'Frames Sent to SmolVLM: {frame_numbers}\nPeople: {colors_str}',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.show()

        print(f"   👁️  Displayed {num_frames} frames above")
        print(f"{'='*70}\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_multi_person_tracking(video_path, video_detections, tracked_people,
                                   output_path='multi_person_tap_output.mp4',
                                   initial_frame=0):
    """Create video with color-coded multi-person tracking"""
    try:
        import supervision as sv
    except ImportError:
        print("⚠️  Installing supervision...")
        import subprocess
        subprocess.check_call(["pip", "install", "supervision>=0.26.0"])
        import supervision as sv

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

    frame_idx = initial_frame
    max_frame = max(video_detections.keys()) if video_detections else initial_frame

    print("🎬 Creating output video with color-coded boxes...")

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

                    # Use person's assigned color
                    color = person.color

                    # Draw thick colored box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Status label
                    if person.has_tapped and frame_idx >= person.tap_frame:
                        status = "PAID ✓"
                        bg_color = (0, 200, 0)  # Green background
                    else:
                        status = "Waiting..."
                        bg_color = (50, 50, 50)  # Dark gray background

                    label = f"{person.color_name}: {status}"
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Background for label
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(frame,
                                (x1, y1 - text_height - 10),
                                (x1 + text_width + 10, y1),
                                bg_color, -1)

                    # White text
                    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                              font, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"   Processed {frame_idx - initial_frame} frames...")

    cap.release()
    out.release()
    print(f"✅ Video saved: {output_path}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example usage:
"""
# 1. Create tracker with RT-DETR model
# tracker = MultiPersonTapTracker(
#     rtdetr_model="rtdetr-x.pt",
#     conf_threshold=0.7,
#     frames_per_check=6
# )

# # 2. Track all people and detect taps with color-coded boxes
# video_detections, tracked_people = tracker.track_all_people(
#     video_path="/content/Bus_Boarding_Fare_Validation_Sequence.mp4",
#     check_interval=30,  # Check cycle every 30 frames (10, 20, 30 pattern)
#     initial_frame=0
# )

# Frame pattern explanation:
# 1st VLM call: frames 10, 20, 30
# 2nd VLM call: frames 40, 50, 60
# 3rd VLM call: frames 70, 80, 90
# No frame is ever sent twice!

# 3. Visualize results with color-coded boxes
# visualize_multi_person_tracking(
#     video_path="/content/Bus_Boarding_Fare_Validation_Sequence.mp4",
#     video_detections=video_detections,
#     tracked_people=tracked_people,
#     output_path="tap_detection_color_coded.mp4",
#     initial_frame=0
# )