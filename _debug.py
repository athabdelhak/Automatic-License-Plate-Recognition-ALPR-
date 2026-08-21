"""Debug raw OCR output on problem images."""
import os, cv2, logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO
from paddleocr import PaddleOCR

model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_thresh=0.15,
    text_det_box_thresh=0.25,
    text_det_unclip_ratio=2.5,
)

for img_name in ("car1.jpg", "car2.jpg", "car4.jpg", "car6.jpg"):
    p = os.path.join(SCRIPT_DIR, "test_images", img_name)
    frame = cv2.imread(p)
    print(f"\n=== {img_name}  shape={frame.shape} ===")
    res = model(frame, verbose=False, conf=0.05)
    for r in res:
        if r.boxes is None: continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            print(f"  yolo box {i} conf={float(box.conf[0]):.2f} crop={crop.shape}")
            cv2.imwrite(os.path.join(SCRIPT_DIR, f"_dbg_{img_name}_{i}.png"), crop)
            for label, img in [("orig", crop),
                               ("3x", cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)),
                               ("gray", cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)),
                               ]:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                out = ocr.predict(img)
                if not out: print(f"    [{label}] no result"); continue
                page = out[0]
                texts = page.get("rec_texts", []) or []
                scores = page.get("rec_scores", []) or []
                print(f"    [{label}] texts={texts}")
                print(f"            scores={[round(float(s),2) for s in scores]}")
    # Also try OCR on the WHOLE frame in case YOLO crop is off
    out = ocr.predict(frame)
    if out:
        page = out[0]
        texts = page.get("rec_texts", []) or []
        scores = page.get("rec_scores", []) or []
        print(f"  [whole frame] texts={texts}")
        print(f"  [whole frame] scores={[round(float(s),2) for s in scores]}")
