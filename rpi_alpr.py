"""
rpi_alpr.py

Pipeline (on Raspberry Pi 5 + Pi Camera v1.3):
  Camera -> YOLO finds the plate region -> PaddleOCR reads the text
         -> Vote on the best read -> Publish over MQTT to the laptop

A live preview window shows the camera feed with the YOLO box drawn on it.
"""

# ---- Standard imports + force unbuffered logs (so output is real-time) ----
import os
import sys
import time
import json
import signal
import logging
import warnings
from collections import deque, Counter

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


# ---- Configuration: edit BROKER_HOST to the laptop's LAN IP ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

BROKER_HOST = "10.44.152.219"
BROKER_PORT = 1883
TOPIC       = "alpr/plate"
CLIENT_ID   = "rpi5-alpr"

FRAME_W, FRAME_H = 1280, 720

SHOW_PREVIEW = True
PREVIEW_SAVE = "/tmp/alpr_preview.jpg"

YOLO_CONF        = 0.5
STABLE_FRAMES    = 3
VOTE_BUFFER      = 5
VOTE_NEEDED      = 1
PUBLISH_COOLDOWN = 5.0


# ---- Logging setup ----
warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alpr")


# ---- Heavy imports: AI libraries and the camera driver ----
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import paho.mqtt.client as mqtt
from picamera2 import Picamera2


# ---- Plate text filtering: rules to keep real plates and drop brand names / URLs ----
NON_PLATE_WORDS = {
    "BMW", "AUDI", "VW", "FORD", "MERCEDES", "BENZ", "OPEL", "RENAULT",
    "PEUGEOT", "FIAT", "TOYOTA", "HONDA", "KIA", "HYUNDAI", "NISSAN",
    "MAZDA", "SKODA", "DACIA", "TESLA", "VOLVO",
    "WWW", "COM", "RO", "FR", "DE", "EU", "UK", "USA", "ORG", "NET",
    "ROMAN", "ROMANIA", "FRANCE", "GERMANY", "GROUP",
}


def normalize(s):
    return "".join(c for c in s if c.isalnum()).upper()


def plate_score(text, ocr_score):
    n = len(text)
    if not (4 <= n <= 11):
        return -1
    if text in NON_PLATE_WORDS:
        return -1
    for w in NON_PLATE_WORDS:
        if len(w) >= 5 and w in text:
            return -1
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    if not has_digit:
        return -1
    if not has_letter and n < 6:
        return -1
    length_bonus = 0.03 * min(n, 10)
    return float(ocr_score) + length_bonus


# ---- OCR helpers: enlarge tiny crops, group OCR boxes into lines, read a plate ----
def upscale(img, target_h=120):
    ih, iw = img.shape[:2]
    if ih >= target_h:
        return img
    s = target_h / ih
    return cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_CUBIC)


def assemble_lines(rec_texts, rec_scores, rec_boxes):
    items = []
    for txt, score, box in zip(rec_texts, rec_scores, rec_boxes):
        cleaned = normalize(txt)
        if not cleaned:
            continue
        x1, y1, x2, y2 = box
        items.append({
            "text": cleaned,
            "score": float(score),
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "h": max(1, y2 - y1),
        })
    if not items:
        return []
    avg_h = sum(it["h"] for it in items) / len(items)
    items.sort(key=lambda it: it["cy"])
    lines = [[items[0]]]
    for it in items[1:]:
        if abs(it["cy"] - lines[-1][-1]["cy"]) < avg_h * 0.7:
            lines[-1].append(it)
        else:
            lines.append([it])
    out = []
    for line in lines:
        line.sort(key=lambda it: it["cx"])
        full = "".join(it["text"] for it in line)
        avg = sum(it["score"] for it in line) / len(line)
        out.append((full, avg))
        for it in line:
            out.append((it["text"], it["score"]))
    return out


def read_plate(crop, ocr):
    best = None
    for variant in (crop, upscale(crop, target_h=120)):
        try:
            res = ocr.ocr(variant, cls=False)
        except Exception as e:
            log.warning("OCR error: %s", str(e).splitlines()[0] if str(e) else type(e).__name__)
            continue
        if not res:
            continue
        page = res[0]
        if not page:
            continue
        texts, scores, boxes = [], [], []
        for entry in page:
            try:
                poly, (txt, score) = entry
            except (ValueError, TypeError):
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            texts.append(txt)
            scores.append(score)
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        if not boxes:
            continue
        for text, oscore in assemble_lines(texts, scores, boxes):
            s = plate_score(text, oscore)
            if s < 0:
                continue
            if best is None or s > best[1]:
                best = (text, s, oscore)
    if best is None:
        return None
    return best[0], best[2]


# ---- MQTT: build the client we use to send plates to the laptop ----
def make_mqtt_client():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except AttributeError:
        client = mqtt.Client(client_id=CLIENT_ID)
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(c, userdata, flags, *args):
        rc = args[0] if args else 0
        if rc == 0:
            log.info("MQTT connected to %s:%s (topic=%s)", BROKER_HOST, BROKER_PORT, TOPIC)
        else:
            log.error("MQTT connect failed, reason_code=%s", rc)

    def on_disconnect(c, userdata, *args):
        log.warning("MQTT disconnected; auto-reconnecting...")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    return client


# ---- Geometry: tells us if two YOLO boxes are the same plate (high overlap) ----
def boxes_overlap(a, b, iou_thresh=0.4):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return False
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union >= iou_thresh


# ---- Main: load models, start camera, connect MQTT, then run the detection loop ----
def main():
    if not os.path.exists(MODEL_PATH):
        log.error("YOLO weights not found at %s", MODEL_PATH)
        sys.exit(1)

    # Load the YOLO plate detector and the PaddleOCR text reader.
    log.info("Loading YOLO from %s", MODEL_PATH)
    yolo = YOLO(MODEL_PATH)

    log.info("Loading PaddleOCR (v2.x API, PP-OCRv4 en)...")
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        det_db_thresh=0.2,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=2.0,
        show_log=False,
    )

    # Start the Pi camera and wait for auto-exposure to settle.
    log.info("Starting Pi camera %dx%d...", FRAME_W, FRAME_H)
    cam = Picamera2()
    log.info("Picamera2() ok, configuring...")
    cfg = cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    cam.configure(cfg)
    log.info("Configured, starting camera...")
    cam.start()
    log.info("Camera started, warming up...")
    time.sleep(1.0)
    log.info("Camera ready.")

    # Connect to the MQTT broker on the laptop.
    client = make_mqtt_client()
    client.loop_start()
    log.info("MQTT: trying to connect to %s:%s ...", BROKER_HOST, BROKER_PORT)
    connected_ok = False
    for attempt in range(3):
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
            connected_ok = True
            break
        except Exception as e:
            log.warning("MQTT connect attempt %d failed: %s", attempt + 1, e)
            time.sleep(2)
    if not connected_ok:
        log.error("MQTT: could not reach %s:%s after 3 tries. "
                  "Check that the laptop IP/broker is reachable. "
                  "Continuing anyway, will keep retrying in background.",
                  BROKER_HOST, BROKER_PORT)

    # Clean shutdown on Ctrl+C.
    stop = {"v": False}

    def handle_sig(*_):
        log.info("Signal received, shutting down...")
        stop["v"] = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # State kept across frames: stability counter, vote buffer, publish cooldown.
    stable_count = 0
    last_box = None
    vote_buffer = deque(maxlen=VOTE_BUFFER)
    last_published = {}
    last_plate_text = ""
    last_plate_until = 0

    # Open the live preview window. If no display, save snapshots instead.
    preview_mode = "none"
    if SHOW_PREVIEW:
        try:
            cv2.namedWindow("ALPR preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ALPR preview", 960, 540)
            preview_mode = "window"
            log.info("Preview window open. Press 'q' or Ctrl+C to quit.")
        except Exception as e:
            preview_mode = "file"
            log.warning("No display, saving annotated frames to %s instead (%s)",
                        PREVIEW_SAVE, e)
    save_every_n = 5
    frame_idx = 0

    log.info("Ready. Watching for plates...")

    # ---- Main detection loop: capture, detect, OCR, vote, publish ----
    try:
        while not stop["v"]:
            # Capture a frame and convert RGB -> BGR for OpenCV.
            frame_rgb = cam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_idx += 1

            # YOLO step: keep the highest-confidence plate box, if any.
            results = yolo(frame, verbose=False, conf=YOLO_CONF)
            current_box = None
            current_conf = 0.0
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                best_b = max(r.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best_b.xyxy[0])
                current_box = (x1, y1, x2, y2)
                current_conf = float(best_b.conf[0])

            # Build the live preview (rectangle + status + last plate).
            disp = frame
            if preview_mode != "none":
                disp = frame.copy()
                if current_box is not None:
                    bx1, by1, bx2, by2 = current_box
                    color = (0, 255, 0) if stable_count + 1 >= STABLE_FRAMES else (0, 200, 255)
                    cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(disp, f"plate {current_conf:.2f}",
                                (bx1, max(20, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                status = f"stable {stable_count}/{STABLE_FRAMES}  votes {len(vote_buffer)}/{VOTE_BUFFER}"
                cv2.putText(disp, status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if last_plate_text and time.time() < last_plate_until:
                    cv2.putText(disp, f"PLATE: {last_plate_text}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if preview_mode == "window":
                    cv2.imshow("ALPR preview", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        stop["v"] = True
                elif preview_mode == "file" and frame_idx % save_every_n == 0:
                    try:
                        cv2.imwrite(PREVIEW_SAVE, disp)
                    except Exception:
                        pass

            # Track stability: same plate as last frame, or a new one?
            if current_box is None:
                stable_count = 0
                last_box = None
                vote_buffer.clear()
                continue

            if last_box is not None and boxes_overlap(current_box, last_box):
                stable_count += 1
            else:
                stable_count = 1
                vote_buffer.clear()
            last_box = current_box

            if stable_count < STABLE_FRAMES:
                continue

            # Crop the plate and run OCR on it.
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = current_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            read = read_plate(crop, ocr)
            if read is None:
                log.info("OCR: no readable plate in crop")
                continue
            plate_text, ocr_conf = read
            vote_buffer.append((plate_text, ocr_conf))
            log.info("OCR: %-12s ocr_conf=%.2f  votes=%s",
                     plate_text, ocr_conf, list(vote_buffer))

            # Vote: pick the most frequent recent read; respect cooldown.
            tally = Counter(t for t, _ in vote_buffer)
            top_text, top_count = tally.most_common(1)[0]
            need = max(VOTE_NEEDED, 1)
            if top_count < need:
                continue

            now = time.time()
            if now - last_published.get(top_text, 0) < PUBLISH_COOLDOWN:
                continue

            # Publish the winning plate to MQTT as JSON.
            avg_conf = sum(c for t, c in vote_buffer if t == top_text) / top_count
            payload = json.dumps({
                "plate": top_text,
                "confidence": round(avg_conf, 3),
                "ts": now,
            })
            info = client.publish(TOPIC, payload, qos=1)
            last_published[top_text] = now
            last_plate_text = top_text
            last_plate_until = now + 3.0
            try:
                rc = info.rc
            except AttributeError:
                rc = "?"
            log.info("PLATE: %-12s  conf=%.2f  -> published (mid=%s, rc=%s, connected=%s)",
                     top_text, avg_conf, info.mid, rc, client.is_connected())

    # ---- Cleanup: release camera, close window, disconnect MQTT ----
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass
        log.info("Stopped.")


if __name__ == "__main__":
    main()
