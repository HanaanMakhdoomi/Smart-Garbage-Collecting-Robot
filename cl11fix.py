import os
import cv2
import json
import math
import time
import socket
import threading
import numpy as np
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 1

# One ESP32 does both the car motors and the arm.
# Use the IP printed by its Serial Monitor.
ESP32_IP = "YourEsp32IPaddress"
ESP32_PORT = 5000

# Gemini API Key (set via environment variable or in .env file)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------- Distance calibration (width method) ----------
KNOWN_OBJECT_WIDTH_CM = 8.0
CALIBRATION_DISTANCE_CM = 50.0
CALIBRATION_WIDTH_PX = 120.0

# ---------- Distance calibration (bottom-edge method, optional) ----------
# More accurate for floor objects of any size. Place an object at several
# distances, note the box bottom edge y2 (pixels, 480p frame), fill table.
# MUST be sorted by y2 ascending (far = small y2, near = large y2).
USE_BOTTOM_EDGE_DISTANCE = False
Y2_CALIBRATION = [
    # (y2_px, distance_cm)  <-- REPLACE with your own measurements
    (250, 100.0),
    (330, 80.0),
    (400, 50.0),
    (450, 30.0),
]

# ---------- Stopping (dead reckoning) ----------
# Vision measures the distance while the car is STANDING STILL (right after
# Gemini finds/re-checks the garbage). From then on the program counts how
# far the car has driven (speed x time) and stops after exactly that far.
# This works even when the garbage disappears under the camera at the end.
STOP_DISTANCE_CM = 0.0          # distance left between car and garbage
                                 # when it stops (set to your arm's reach)
CAR_SPEED_CM_PER_S = 39        # <-- MUST MEASURE: press T (see README
                                 #     in the reply): speed at FORWARD_SPEED
CALIBRATION_DRIVE_SECONDS = 2.0  # length of the T test drive

# Final approach: with this little distance left the car "commits": it
# stops steering and turning, ignores the camera and drives straight.
COMMIT_REMAINING_CM = 35.0
COMMIT_MAX_ERROR = 0.25          # must be roughly aimed before committing

# Backup: also stop if the camera says it is much closer than planned.
VISION_BACKUP_STOP = True
VISION_STOP_FACTOR = 0.6         # x STOP_DISTANCE_CM

# Motor PWM: 0 to 255 (1 kHz PWM on the ESP32).
# HARD_PWM_LIMIT is the absolute ceiling: nothing is ever sent above it.
HARD_PWM_LIMIT = 200             # must match MAX_PWM in the ESP32 sketch
FORWARD_SPEED = 100              # constant straight-driving speed
MIN_CONFIDENCE = 0.70
CONTROL_INTERVAL_SECONDS = 0.12

# ---------- Steering ----------
# Error = how far the target is from the aim line, from -1 (far left)
# to +1 (far right).
ALIGN_ENTER = 0.30               # beyond this: stop and turn in place
ALIGN_EXIT = 0.10                # inside this: drive forward again
TURN_SPEED = 150                 # PWM for turning in place. If it turns
                                 # too far per pulse, lower it; if the
                                 # motors hum but it won't rotate, raise it
                                 # (up to HARD_PWM_LIMIT).
TURN_PULSE_MIN_SECONDS = 0.08    # shortest turn burst (target nearly centred)
TURN_PULSE_MAX_SECONDS = 0.20    # longest turn burst (target at the edge)
TURN_PAUSE_SECONDS = 0.70        # wait for the laggy camera to show the result
TURN_DIRECTION = 1               # set to -1 if it turns AWAY from the target
STALL_PULSES = 4                 # pulses without improvement = stalled
STALL_MIN_IMPROVEMENT = 0.03     # how much the error must shrink
MAX_STALLS = 2                   # stalls before the lock is released
STEER_GAIN = 40                  # correction while driving forward
STEER_MAX_BOOST = 12             # max extra PWM on the outer wheel
AIM_X_OFFSET_PX = 0              # shift if the camera is not centred on the
                                 # robot (+ moves the aim line right)

# ---------- Target lock / tracking ----------
SCAN_INTERVAL_SECONDS = 3.0      # Gemini scans while no target is locked
SETTLE_SECONDS = 0.5             # stand still before a verification scan
RESCAN_DELAY_SECONDS = 1.0
VERIFY_INTERVAL_SECONDS = 8.0    # re-check the locked target (0 = disable)
MATCH_RADIUS_PX = 120            # a scan "matches" if within this of the lock
MAX_MISSES = 3                   # failed re-acquisitions before giving up
LOST_FRAME_LIMIT = 8             # tracker failures (frames) before "lost"
MAX_AREA_JUMP = 3.0              # reject tracker boxes that jump in size
DISTANCE_SMOOTHING = 0.4         # 0..1, higher = faster but noisier
AUTO_RESUME_AFTER_REACH_SECONDS = 0   # 0 = stay stopped until you press R
MAX_DETECTIONS = 5

# ---------- Robotic arm (same ESP32, same connection) ----------
ARM_ENABLED = True
ARM_TIMEOUT_SECONDS = 45.0       # stop waiting for the arm after this long
AUTO_RESUME_AFTER_ARM_SECONDS = 3.0   # after the pick, look for the next
                                      # garbage (0 = wait until you press R)

# =========================================================
# INITIALIZATION
# =========================================================

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. In PowerShell, run:\n"
        '$env:GEMINI_API_KEY = "YOUR_NEW_API_KEY"\n'
        'or create a .env file with GEMINI_API_KEY=YOUR_KEY'
    )

if FORWARD_SPEED > HARD_PWM_LIMIT or TURN_SPEED > HARD_PWM_LIMIT:
    raise ValueError(
        "FORWARD_SPEED/TURN_SPEED exceed HARD_PWM_LIMIT. To go faster, raise "
        "HARD_PWM_LIMIT here AND MAX_PWM in the ESP32 sketch."
    )

if KNOWN_OBJECT_WIDTH_CM <= 0 or CALIBRATION_WIDTH_PX <= 0:
    raise ValueError("Camera calibration values must be positive.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

FOCAL_LENGTH_PX = (
    CALIBRATION_WIDTH_PX * CALIBRATION_DISTANCE_CM / KNOWN_OBJECT_WIDTH_CM
)

# =========================================================
# SHARED STATE (only the Gemini worker hand-off)
# =========================================================

state_lock = threading.Lock()

state = {
    "busy": False,
    "scan_result": None,
}

stop_event = threading.Event()

# =========================================================
# CAMERA / TRACKER / GEOMETRY
# =========================================================

def open_camera(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera index {index}. Check DroidCam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        camera.read()

    return camera


def create_tracker():
    """Create an OpenCV tracker, trying the best available type."""
    names = ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create")

    for module in (cv2, getattr(cv2, "legacy", None)):
        if module is None:
            continue

        for name in names:
            factory = getattr(module, name, None)

            if factory is not None:
                return factory()

    raise RuntimeError(
        "No OpenCV tracker available. Run:\n"
        "pip install opencv-contrib-python"
    )


def box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def measure_distance_cm(box):
    """Estimate ground distance from a (x1, y1, x2, y2) box."""
    if USE_BOTTOM_EDGE_DISTANCE:
        ys = [p[0] for p in Y2_CALIBRATION]
        ds = [p[1] for p in Y2_CALIBRATION]
        return float(np.interp(box[3], ys, ds))

    width = box[2] - box[0]

    if width <= 0:
        return None

    return KNOWN_OBJECT_WIDTH_CM * FOCAL_LENGTH_PX / width


def find_match(detections, ref_box):
    """Return the detection closest to ref_box, or None if none is near."""
    rcx, rcy = box_center(ref_box)
    best, best_dist = None, None

    for det in detections:
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        cx, cy = box_center(box)
        dist = math.hypot(cx - rcx, cy - rcy)

        if dist <= MATCH_RADIUS_PX or box_iou(box, ref_box) > 0.1:
            if best is None or dist < best_dist:
                best, best_dist = det, dist

    return best

# =========================================================
# GEMINI DETECTION
# =========================================================

def parse_gemini_json(response_text):
    if not response_text or not response_text.strip():
        raise ValueError("Gemini returned an empty response.")

    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().lower() in (
            "```", "```json", "```javascript"
        ):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace >= 0 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")

    return result


def parse_detection(item, width, height):
    """Validate one detection dict; return pixel box dict or None."""
    if not isinstance(item, dict):
        return None

    try:
        confidence = float(item.get("confidence", 0))
    except (ValueError, TypeError):
        return None

    if not MIN_CONFIDENCE <= confidence <= 1:
        return None

    bbox = item.get("bbox_1000")

    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x1n, y1n, x2n, y2n = map(float, bbox)
    except (ValueError, TypeError):
        return None

    if not all(
        np.isfinite(v) and 0 <= v <= 1000
        for v in (x1n, y1n, x2n, y2n)
    ):
        return None

    if x2n <= x1n or y2n <= y1n:
        return None

    x1 = max(0, min(width - 1, int(x1n * width / 1000)))
    y1 = max(0, min(height - 1, int(y1n * height / 1000)))
    x2 = max(0, min(width, int(x2n * width / 1000)))
    y2 = max(0, min(height, int(y2n * height / 1000)))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": confidence,
        "label": str(item.get("label") or "garbage"),
    }


def detect_garbage_with_gemini(frame):
    """Return a list of garbage detections (possibly empty)."""
    height, width = frame.shape[:2]

    success, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    if not success:
        raise RuntimeError("Could not encode camera frame.")

    prompt = f"""
You are the vision component of a small indoor garbage-collecting robot.

Find every piece of clearly discarded garbage on the floor in front of
the robot: paper, wrappers, bottles, cans, cups, and similar waste.

Do not list furniture, walls, shoes, or ordinary household objects
as garbage unless they clearly appear to be discarded waste.

Return ONLY valid JSON in this format:
{{
  "detections": [
    {{
      "label": "paper wrapper",
      "confidence": 0.92,
      "bbox_1000": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- List at most {MAX_DETECTIONS} items, one entry per separate object.
- bbox_1000 uses normalized coordinates from 0 to 1000.
- Coordinates are [left, top, right, bottom].
- Origin is the top-left corner.
- The box must tightly enclose the whole object.
- Confidence must be between 0 and 1.
- If no garbage is clearly visible, return {{"detections": []}}.

Image dimensions: {width} x {height}.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=encoded.tobytes(),
                mime_type="image/jpeg",
            ),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            max_output_tokens=1024,
        ),
    )

    response_text = response.text

    if not response_text or not response_text.strip():
        candidates = getattr(response, "candidates", None) or []

        if candidates:
            print(
                "[GEMINI DEBUG] finish_reason=",
                getattr(candidates[0], "finish_reason", None),
            )

        raise RuntimeError("Gemini returned an empty response.")

    result = parse_gemini_json(response_text)
    raw = result.get("detections")

    # Tolerate the old single-object format.
    if raw is None and result.get("garbage_found") is True:
        raw = [result]

    if not isinstance(raw, list):
        return []

    detections = []

    for item in raw[:MAX_DETECTIONS]:
        det = parse_detection(item, width, height)

        if det is not None:
            detections.append(det)

    return detections


def inference_worker(frame_copy, epoch):
    result = {"epoch": epoch, "detections": [], "error": None}

    try:
        result["detections"] = detect_garbage_with_gemini(frame_copy)
        print(f"[GEMINI] {len(result['detections'])} detection(s)")

    except Exception as error:
        result["error"] = str(error)
        print("[ERROR] Gemini detection failed:", error)

    finally:
        with state_lock:
            state["scan_result"] = result
            state["busy"] = False

# =========================================================
# ESP32 TCP COMMUNICATION (motors + arm, one connection)
# =========================================================

sock = None
socket_lock = threading.Lock()


def connect_esp32():
    global sock

    connection = socket.create_connection(
        (ESP32_IP, ESP32_PORT),
        timeout=3.0,
    )
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(1.0)

    with socket_lock:
        sock = connection

    print(f"Connected to ESP32: {ESP32_IP}:{ESP32_PORT}")
    send_motor_command(0, 0)


def send_line(text):
    """Send any newline-terminated command to the ESP32."""
    with socket_lock:
        if sock is None:
            raise ConnectionError("ESP32 socket is unavailable.")

        sock.sendall((text + "\n").encode("ascii"))


def send_motor_command(left, right):
    left = int(np.clip(left, -HARD_PWM_LIMIT, HARD_PWM_LIMIT))
    right = int(np.clip(right, -HARD_PWM_LIMIT, HARD_PWM_LIMIT))

    send_line(f"M,{left},{right}")


def safe_stop():
    try:
        send_motor_command(0, 0)
    except (OSError, ConnectionError):
        pass


def esp_reader(arm):
    """Reads replies (READY, STEP, DONE, ERR...) from the ESP32."""
    buffer = b""

    while not stop_event.is_set():
        with socket_lock:
            s = sock

        if s is None:
            break

        try:
            data = s.recv(256)
        except socket.timeout:
            continue
        except OSError:
            break

        if not data:
            break

        buffer += data

        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            arm._handle(line.decode("ascii", "ignore").strip())

    with arm.lock:
        if arm.state != "OFFLINE" and not stop_event.is_set():
            print("[ESP] Connection lost.")

        arm.state = "OFFLINE"

# =========================================================
# ARM STATE (commands go through the shared ESP32 connection)
# =========================================================

class ArmClient:
    """
    Python only says PICK; the ESP32 runs the whole movement sequence
    itself and answers DONE when it has finished.

    state: OFFLINE | IDLE | RUNNING | DONE | ERROR
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "IDLE"
        self.detail = ""

    def _handle(self, line):
        if not line:
            return

        print("[ESP]", line)

        with self.lock:
            if line == "DONE":
                self.state = "DONE"
                self.detail = "Pick complete"
            elif line.startswith("STEP"):
                self.detail = line
            elif line.startswith("ERR"):
                self.state = "ERROR"
                self.detail = line
            elif line == "ABORTED":
                self.state = "IDLE"

    def start_pick(self):
        with self.lock:
            if self.state in ("OFFLINE", "RUNNING"):
                return False

        try:
            send_line("PICK")
        except (OSError, ConnectionError):
            with self.lock:
                self.state = "OFFLINE"

            return False

        with self.lock:
            self.state = "RUNNING"
            self.detail = "Starting"

        return True

    def snapshot(self):
        with self.lock:
            return self.state, self.detail

    def close(self):
        pass

# =========================================================
# PILOT: turns the target error into motor commands
# =========================================================

class Pilot:
    """Turn in short pulses until aligned, then drive forward and steer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.aligned = False
        self.pulse_until = 0.0
        self.pause_until = 0.0
        self.best_error = float("inf")
        self.stall_pulses = 0
        self.stalled = False

    @staticmethod
    def forward_speed(distance_cm):
        # One constant speed, so the dead-reckoning calibration stays valid.
        return FORWARD_SPEED

    def command(self, now, distance_cm, error):
        abs_error = abs(error)

        # Hysteresis so it doesn't flip between turning and driving.
        if self.aligned and abs_error > ALIGN_ENTER:
            self.aligned = False
        elif not self.aligned and abs_error < ALIGN_EXIT:
            self.aligned = True

        if not self.aligned:
            direction = (1 if error > 0 else -1) * TURN_DIRECTION
            turn = (direction * TURN_SPEED, -direction * TURN_SPEED)

            if now < self.pulse_until:
                return turn, "TURNING"

            if now < self.pause_until:
                return (0, 0), "WAITING FOR CAMERA"

            # Start a new pulse. First check the last ones actually helped;
            # otherwise the robot would spin forever.
            if abs_error < self.best_error - STALL_MIN_IMPROVEMENT:
                self.best_error = abs_error
                self.stall_pulses = 0
            else:
                self.stall_pulses += 1

            if self.stall_pulses >= STALL_PULSES:
                self.stalled = True
                return (0, 0), "TURN STALLED"

            # Bigger error -> longer pulse, small error -> tiny nudge.
            pulse = TURN_PULSE_MIN_SECONDS + (
                TURN_PULSE_MAX_SECONDS - TURN_PULSE_MIN_SECONDS
            ) * min(1.0, abs_error)

            self.pulse_until = now + pulse
            self.pause_until = self.pulse_until + TURN_PAUSE_SECONDS
            return turn, "TURNING"

        # Aligned: drive forward.
        self.best_error = float("inf")
        self.stall_pulses = 0

        base = self.forward_speed(distance_cm)

        # The speed range is narrow, so slowing the inner wheel would stop
        # it. Instead the inner wheel keeps the base speed and only the
        # outer wheel speeds up a little (target right -> left wheel faster).
        boost = float(np.clip(
            abs(error) * STEER_GAIN, 0, STEER_MAX_BOOST
        ))

        left = base + (boost if error > 0 else 0)
        right = base + (boost if error < 0 else 0)

        left = int(np.clip(left, 0, HARD_PWM_LIMIT))
        right = int(np.clip(right, 0, HARD_PWM_LIMIT))

        return (left, right), "DRIVING"

# =========================================================
# ROBOT: target locking + state machine
# =========================================================

class Robot:
    """
    Modes:
      SEARCHING : no target; stopped; Gemini scans periodically.
      LOCKED    : one target is locked and followed by a tracker.
      VERIFYING : stopped; Gemini re-checks the SAME target.
      REACHED   : stopped at the target.

    Once locked, other garbage is ignored. The lock is only released
    if the target is lost, reached (and you press R), or you press R.
    """

    def __init__(self, arm=None):
        self.arm = arm
        self.arm_started = False
        self.arm_started_time = 0.0
        self.arm_done_time = None

        self.mode = "SEARCHING"
        self.status = "STARTING"
        self.message = "Waiting for first scan"

        self.tracker = None
        self.bbox = None
        self.label = ""
        self.confidence = 0.0
        self.distance_cm = None

        self.lost_frames = 0
        self.miss_count = 0
        self.tracker_lost = False

        self.epoch = 0
        self.next_scan_time = 0.0
        self.next_verify_time = 0.0
        self.verify_settle_until = 0.0
        self.scan_started = False
        self.reached_time = 0.0

        self.stall_count = 0
        self.pilot = Pilot()

        # Dead reckoning
        self.remaining_cm = None   # distance still to drive before stopping
        self.committed = False     # final approach: straight, no steering
        self.prev_cmd = (0, 0)
        self.last_time = None

    # ---------- Gemini hand-off ----------

    @staticmethod
    def is_busy():
        with state_lock:
            return state["busy"]

    @staticmethod
    def take_scan_result():
        with state_lock:
            result = state["scan_result"]
            state["scan_result"] = None

        return result

    def start_scan(self, frame):
        with state_lock:
            if state["busy"]:
                return False

            state["busy"] = True

        threading.Thread(
            target=inference_worker,
            args=(frame.copy(), self.epoch),
            daemon=True,
        ).start()

        return True

    # ---------- Lock management ----------

    def lock_target(self, det, frame, now, fresh=True):
        tracker = create_tracker()

        x, y = det["x1"], det["y1"]
        w, h = det["x2"] - det["x1"], det["y2"] - det["y1"]

        try:
            ok = tracker.init(frame, (x, y, w, h))
        except cv2.error as error:
            print("[TRACKER] init failed:", error)
            return False

        if ok is False:
            return False

        self.tracker = tracker
        self.bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
        self.label = det["label"]
        self.confidence = det["confidence"]
        self.distance_cm = measure_distance_cm(self.bbox)

        self.lost_frames = 0
        self.miss_count = 0
        self.tracker_lost = False

        # The car is standing still here, so this is a trustworthy fix:
        # (re)start the dead-reckoning countdown from it.
        self.remaining_cm = (
            self.distance_cm - STOP_DISTANCE_CM
            if self.distance_cm is not None else None
        )

        if fresh:
            # A new target: start aiming from scratch. A re-check of the
            # same target keeps the current alignment state.
            self.pilot.reset()
            self.committed = False

        self.next_verify_time = now + VERIFY_INTERVAL_SECONDS
        self.mode = "LOCKED"
        self.status = "LOCKED"
        self.message = f"{self.label} | {self.distance_cm:.0f} cm"

        print(f"[LOCK] {self.label} at {self.distance_cm:.0f} cm")
        return True

    def release_lock(self, now, reason):
        self.tracker = None
        self.bbox = None
        self.distance_cm = None
        self.tracker_lost = False
        self.scan_started = False

        self.epoch += 1          # discard any scan still in flight
        self.stall_count = 0
        self.arm_started = False
        self.arm_done_time = None
        self.remaining_cm = None
        self.committed = False
        self.mode = "SEARCHING"
        self.next_scan_time = now + 1.0
        self.pilot.reset()

        self.status = "SEARCHING"
        self.message = reason
        print("[RELEASE]", reason)

    def request_scan(self, now):
        if self.mode == "SEARCHING":
            self.next_scan_time = now
        elif self.mode == "LOCKED":
            self.next_verify_time = now

    def enter_verify(self, now, lost):
        self.mode = "VERIFYING"
        self.tracker_lost = lost
        self.scan_started = False
        self.verify_settle_until = now + SETTLE_SECONDS
        self.miss_count = 0 if lost else self.miss_count
        self.status = "RE-ACQUIRING" if lost else "VERIFYING"
        self.message = "Stopped; checking the locked target"

    # ---------- Tracker ----------

    def update_tracker(self, frame):
        height, width = frame.shape[:2]
        ok, box = False, None

        try:
            ok, box = self.tracker.update(frame)
        except cv2.error as error:
            print("[TRACKER] update failed:", error)

        if ok:
            x, y, w, h = box
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)

            new_w, new_h = x2 - x1, y2 - y1

            # Box mostly outside the frame = target left the view.
            leaving = w > 0 and new_w < 0.5 * w
            old_area = max(
                1.0,
                (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]),
            )
            ratio = (new_w * new_h) / old_area

            valid = (
                not leaving
                and new_w >= 4
                and new_h >= 4
                and 1.0 / MAX_AREA_JUMP <= ratio <= MAX_AREA_JUMP
            )
        else:
            valid = False

        if not valid:
            self.lost_frames += 1
            return

        self.lost_frames = 0
        self.bbox = (x1, y1, x2, y2)

        new_distance = measure_distance_cm(self.bbox)

        if new_distance is not None:
            if self.distance_cm is None:
                self.distance_cm = new_distance
            else:
                self.distance_cm = (
                    DISTANCE_SMOOTHING * new_distance
                    + (1.0 - DISTANCE_SMOOTHING) * self.distance_cm
                )

    # ---------- Scan results ----------

    def handle_scan_result(self, result, frame, now):
        if self.mode == "SEARCHING":
            if result["error"]:
                self.status = "API ERROR"
                self.message = result["error"][:100]
            elif not result["detections"]:
                self.status = "NO GARBAGE"
                self.message = "No confident garbage detected"
            else:
                # Pick the nearest piece (lowest in the image) and lock it.
                target = max(
                    result["detections"],
                    key=lambda d: (d["y2"], d["confidence"]),
                )

                if self.lock_target(target, frame, now):
                    return

                self.status = "TRACKER ERROR"
                self.message = "Could not start tracker"

            self.next_scan_time = now + SCAN_INTERVAL_SECONDS

        elif self.mode == "VERIFYING":
            match = None

            if not result["error"]:
                match = find_match(result["detections"], self.bbox)

            if match is not None and self.lock_target(
                match, frame, now, fresh=False
            ):
                self.status = "LOCKED (verified)"
                return

            self.miss_count += 1
            limit = MAX_MISSES if self.tracker_lost else 2

            if self.miss_count >= limit:
                self.release_lock(now, "Target lost; searching again")
            elif self.tracker_lost:
                # Try again shortly.
                self.scan_started = False
                self.verify_settle_until = now + RESCAN_DELAY_SECONDS
            else:
                # Tracker still healthy; carry on, check again later.
                self.mode = "LOCKED"
                self.status = "LOCKED"
                self.message = "Verify scan did not confirm target"
                self.next_verify_time = now + VERIFY_INTERVAL_SECONDS

    # ---------- Reaching the target / arm ----------

    def enter_reached(self, now, message):
        self.mode = "REACHED"
        self.reached_time = now
        self.arm_started = False
        self.arm_done_time = None
        self.status = "TARGET REACHED"
        self.message = message
        print("[REACHED]", message)

    def arm_available(self):
        return (
            self.arm is not None
            and self.arm.snapshot()[0] != "OFFLINE"
        )

    def handle_reached(self, now):
        """Runs every frame while the car is stopped at the target."""
        if not self.arm_available():
            if (
                AUTO_RESUME_AFTER_REACH_SECONDS > 0
                and now - self.reached_time > AUTO_RESUME_AFTER_REACH_SECONDS
            ):
                self.release_lock(now, "Looking for the next garbage")

            return

        arm_state, detail = self.arm.snapshot()

        if not self.arm_started:
            if arm_state == "RUNNING":
                self.status = "WAITING FOR ARM"
                self.message = "Arm is still busy with the last pick"
            elif self.arm.start_pick():
                self.arm_started = True
                self.arm_started_time = now
                self.status = "ARM PICKING"
                self.message = "Arm sequence started"

            return

        if arm_state == "DONE":
            if self.arm_done_time is None:
                self.arm_done_time = now
                self.status = "PICK COMPLETE"
                self.message = "Arm finished; car stays still"

            if (
                AUTO_RESUME_AFTER_ARM_SECONDS > 0
                and now - self.arm_done_time > AUTO_RESUME_AFTER_ARM_SECONDS
            ):
                self.release_lock(now, "Pick complete; looking for next garbage")

        elif arm_state == "ERROR":
            self.status = "ARM ERROR"
            self.message = detail[:90]

        elif arm_state == "OFFLINE":
            self.status = "ARM OFFLINE"
            self.message = "Lost connection to the arm; press R"

        elif now - self.arm_started_time > ARM_TIMEOUT_SECONDS:
            self.status = "ARM TIMEOUT"
            self.message = "No DONE from the arm; press R to continue"

        else:
            self.status = "ARM PICKING"
            self.message = detail[:90]

    # ---------- One step of the state machine ----------

    def integrate_travel(self, now):
        """Dead reckoning: count how far the car drove since the last frame."""
        if self.last_time is None:
            dt = 0.0
        else:
            dt = min(now - self.last_time, 0.25)

        self.last_time = now

        left, right = self.prev_cmd

        if (
            self.remaining_cm is not None
            and left > 0 and right > 0
            and dt > 0
        ):
            factor = ((left + right) / 2.0) / FORWARD_SPEED
            self.remaining_cm -= CAR_SPEED_CM_PER_S * factor * dt

    def step(self, frame, now):
        """Process one camera frame and return the motor command."""
        self.integrate_travel(now)
        command = self._step(frame, now)
        self.prev_cmd = (command[0], command[1])
        return command

    def _step(self, frame, now):
        height, width = frame.shape[:2]

        result = self.take_scan_result()

        if result is not None and result["epoch"] == self.epoch:
            self.handle_scan_result(result, frame, now)

        if self.mode == "SEARCHING":
            if not self.is_busy() and now >= self.next_scan_time:
                if self.start_scan(frame):
                    self.status = "ANALYZING"
                    self.message = "Gemini is looking for garbage"
                    self.next_scan_time = now + 3600.0  # until result

            return 0, 0

        if self.mode == "VERIFYING":
            if (
                not self.scan_started
                and now >= self.verify_settle_until
                and not self.is_busy()
            ):
                if self.start_scan(frame):
                    self.scan_started = True

            return 0, 0

        if self.mode == "REACHED":
            self.handle_reached(now)
            return 0, 0

        # ---------------- LOCKED ----------------
        self.update_tracker(frame)
        tracker_ok = self.lost_frames == 0

        # 1) Stop: the dead-reckoning countdown reached zero...
        reached = self.remaining_cm is None or self.remaining_cm <= 0

        # ...or (backup) the camera says it is much closer than planned.
        if (
            not reached
            and tracker_ok
            and VISION_BACKUP_STOP
            and self.distance_cm is not None
            and self.distance_cm <= STOP_DISTANCE_CM * VISION_STOP_FACTOR
        ):
            reached = True

        if reached:
            self.enter_reached(now, "Stopped at the target distance")
            return 0, 0

        error = None

        if tracker_ok:
            center_x = (self.bbox[0] + self.bbox[2]) / 2.0
            aim_x = width / 2.0 + AIM_X_OFFSET_PX
            error = float(
                np.clip((center_x - aim_x) / (width / 2.0), -1.0, 1.0)
            )

        # 2) Final approach: once roughly aimed (or once the garbage has
        #    slipped out of view), just drive straight until the countdown
        #    ends. No more turning, no more verification.
        if not self.committed and self.remaining_cm <= COMMIT_REMAINING_CM:
            if error is None or abs(error) <= COMMIT_MAX_ERROR:
                self.committed = True
                print(
                    f"[FINAL] Committed; {self.remaining_cm:.0f} cm to go "
                    "by dead reckoning"
                )

        if self.committed:
            self.status = "FINAL APPROACH"
            self.message = f"Driving straight | {self.remaining_cm:.0f} cm left"
            return FORWARD_SPEED, FORWARD_SPEED

        # 3) Tracker trouble (garbage still far away).
        if not tracker_ok:
            if self.lost_frames >= LOST_FRAME_LIMIT:
                self.enter_verify(now, lost=True)

            return 0, 0

        # 4) Periodic re-check (only while there is still a long way to go).
        if (
            VERIFY_INTERVAL_SECONDS > 0
            and now >= self.next_verify_time
            and not self.is_busy()
            and self.remaining_cm > COMMIT_REMAINING_CM + 15.0
        ):
            self.enter_verify(now, lost=False)
            return 0, 0

        # 5) Aim at the target and drive.
        command, text = self.pilot.command(now, self.distance_cm, error)

        if self.pilot.stalled:
            self.stall_count += 1
            print(
                "[STALL] Turning is not centring the target. Check "
                "TURN_DIRECTION (turns away?), TURN_SPEED (too weak?) "
                "and that the tracker is on the right object."
            )

            if self.stall_count >= MAX_STALLS:
                self.release_lock(now, "Turn stalled; releasing target")
            else:
                self.enter_verify(now, lost=False)

            return 0, 0

        if text == "DRIVING":
            self.stall_count = 0

        self.status = text
        self.message = (
            f"{self.label} | {self.remaining_cm:.0f} cm left | "
            f"err={error:+.2f} | L={command[0]} R={command[1]}"
        )

        return command

    # ---------- Display ----------

    def draw(self, frame):
        display = frame.copy()
        height, width = display.shape[:2]
        aim_x = int(width / 2 + AIM_X_OFFSET_PX)

        cv2.line(display, (aim_x, 0), (aim_x, height), (120, 120, 120), 1)

        if self.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in self.bbox)

            color = {
                "LOCKED": (0, 255, 0),
                "VERIFYING": (0, 255, 255),
                "REACHED": (255, 160, 0),
            }.get(self.mode, (0, 255, 0))

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if self.distance_cm is not None:
                left_txt = (
                    f" | {max(self.remaining_cm, 0):.0f} cm to go"
                    if self.remaining_cm is not None else ""
                )
                label = (
                    f"LOCK: {self.label} {self.confidence:.0%} | "
                    f"est {self.distance_cm:.0f} cm{left_txt} | y2={y2}"
                )
                cv2.putText(
                    display, label,
                    (max(8, x1), max(80, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        busy = self.is_busy()

        cv2.rectangle(display, (0, 0), (width, 65), (25, 25, 25), -1)

        cv2.putText(
            display,
            f"GARBAGE ROBOT | {self.mode} | {self.status}"
            + ("..." if busy else ""),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        cv2.putText(
            display, self.message[:95], (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
        )

        cv2.putText(
            display, "SPACE scan | R release | P arm | T test drive | Q quit",
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        return display

# =========================================================
# MAIN
# =========================================================

def main():
    camera = None
    arm = None
    robot = Robot()

    try:
        print("Starting DroidCam...")
        camera = open_camera(CAMERA_INDEX)

        print("Connecting to the ESP32 (motors + arm)...")
        connect_esp32()
        safe_stop()

        if ARM_ENABLED:
            arm = ArmClient()
            robot.arm = arm
            threading.Thread(
                target=esp_reader, args=(arm,), daemon=True
            ).start()

        now0 = time.monotonic()
        robot.next_scan_time = now0 + 1.0
        next_control_time = now0
        motor_command = (0, 0)
        last_printed = None
        last_sent = None
        calib_until = None

        print("Robot started.")
        print("SPACE: scan/verify | R: release | P: test arm | T: test drive | Q/ESC: quit")

        while not stop_event.is_set():
            ok, frame = camera.read()

            if not ok or frame is None:
                safe_stop()
                print("[WARN] Camera frame unavailable.")
                time.sleep(0.02)
                continue

            now = time.monotonic()

            motor_command = robot.step(frame, now)

            # T: straight test drive to measure CAR_SPEED_CM_PER_S.
            if calib_until is not None:
                if robot.mode != "SEARCHING":
                    calib_until = None
                elif now < calib_until:
                    motor_command = (FORWARD_SPEED, FORWARD_SPEED)
                else:
                    calib_until = None
                    motor_command = (0, 0)
                    print(
                        "[CALIBRATION] Stopped. Measure how far the car "
                        f"travelled (cm), then set CAR_SPEED_CM_PER_S = "
                        f"cm / {CALIBRATION_DRIVE_SECONDS:.1f}"
                    )

            # Send on the regular tick, and immediately whenever the command
            # changes so short turn pulses start and end on time.
            if now >= next_control_time or motor_command != last_sent:
                next_control_time = now + CONTROL_INTERVAL_SECONDS
                last_sent = motor_command

                if motor_command != last_printed:
                    print(f"[MOTOR] L={motor_command[0]} R={motor_command[1]}")
                    last_printed = motor_command

                try:
                    send_motor_command(*motor_command)
                except (OSError, ConnectionError) as error:
                    print("[ESP32 ERROR]", error)
                    safe_stop()
                    raise

            cv2.imshow("Gemini Garbage Robot", robot.draw(frame))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                robot.request_scan(time.monotonic())

            elif key == ord("r"):
                safe_stop()
                robot.release_lock(time.monotonic(), "Released by user")

            elif (
                key == ord("t")
                and robot.mode == "SEARCHING"
                and calib_until is None
            ):
                calib_until = time.monotonic() + CALIBRATION_DRIVE_SECONDS
                print(
                    f"[CALIBRATION] Driving straight for "
                    f"{CALIBRATION_DRIVE_SECONDS:.1f} s at PWM {FORWARD_SPEED}"
                )

            elif key == ord("p") and arm is not None:
                print("[ARM] Manual test: sending PICK")
                arm.start_pick()

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")

    except Exception as error:
        print("[FATAL]", error)

    finally:
        stop_event.set()

        for _ in range(3):
            safe_stop()
            time.sleep(0.03)

        if arm is not None:
            arm.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        global sock

        with socket_lock:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

                sock = None

        print("[EXIT] Program closed; stop command sent where possible.")


if __name__ == "__main__":
    main()
