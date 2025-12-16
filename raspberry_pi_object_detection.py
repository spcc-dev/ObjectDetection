# raspberry_pi_object_detection.py

import cv2
import numpy as np
from pathlib import Path

# ----- YOLOv8 for general object detection -----
from ultralytics import YOLO

# ----- TFLite for numbers and letters -----
#import tensorflow as tf

import easyocr

import pyttsx3
import threading
import queue

speech_queue = queue.Queue()

engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# ------------------ CONFIG -------------------
YOLO_MODEL_PATH = "yolov8n_int8_openvino_model"#"yolov8n.pt"
#TFLITE_DIGIT_MODEL = "crnn_float16.tflite"#"mnist_model.tflite"#"ei-object-detection-object-detection-tensorflow-lite-float32-model.3.lite"#"mnist_model.tflite"   # 0-9 model
#TFLITE_LETTER_MODEL = "crnn_float16.tflite"  # A–Z model
CONFIDENCE_THRESHOLD = 0.5
CAMERA_ID = 0  # Usually 0 for Pi Camera

# ------------------ LOAD MODELS -------------------
# YOLO
yolo_model = YOLO(YOLO_MODEL_PATH)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], quantize=True)

# TFLite interpreters
#igit_interpreter = tf.lite.Interpreter(model_path=TFLITE_DIGIT_MODEL)
#digit_interpreter.allocate_tensors()

#letter_interpreter = tf.lite.Interpreter(model_path=TFLITE_DIGIT_MODEL)
#letter_interpreter.allocate_tensors()

engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Keep track of what is currently spoken
last_detected = None
object_visible = False

# A–Z mapping
LETTER_MAP = {i: chr(65 + i) for i in range(26)}

# ------------------ CAMERA INIT -------------------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

def speak(text):
    speech_queue.put(text)

OCR_CLASSES = {"label", "sign", "text", "plate"}
ocr_cache = {}
frame_count = 0
MAX_OCR_PER_FRAME = 2

# ------------------ MAIN LOOP -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    ocr_count = 0   # ⬅ reset per frame
    
    frame_count += 1
    if frame_count % 300 == 0:   # every ~10 seconds
        ocr_cache.clear()

    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=0.5, stream=False, verbose=False, imgsz=320, save=False,
    show=False)
    
    detected_texts = []
    detected_names = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # print("Detected:", r.names[cls_id])
            label = r.names[cls_id].lower()

            # Draw YOLO bounding box
            # First draw thick white rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 6)

            # Then draw the original green rectangle inside it
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ---- Add text with green background and white text ----
            label_text = f"{label} {conf:.2f}"

            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw green background rectangle
            cv2.rectangle(frame, 
                        (x1, y1 - text_h - 10),
                        (x1 + text_w + 4, y1),
                        (0, 255, 0), 
                        -1)

            # Draw white text on top
            cv2.putText(frame, label_text, 
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 2)

            detected_names.append(label)

            # ---- HARD CAP OCR CALLS PER FRAME ----
            if ocr_count >= MAX_OCR_PER_FRAME:
                continue   # skip OCR for this box
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # avoid crashes
            
            h, w = crop.shape[:2]
            if max(h, w) > 320:
                scale = 320 / max(h, w)
                crop = cv2.resize(crop, None, fx=scale, fy=scale)

             # Read text in the detected box
            # if label not in OCR_CLASSES:
            #     print('LABEL NOT IN CLASSES')
            #     continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            # text_results = reader.readtext(
            #     gray,
            #     detail=0,      # returns only text (faster)
            #     paragraph=False
            # )

            # for (_, text, _) in text_results:
            #     detected_texts.append(text)
            # ---- CREATE STABLE BOX KEY (quantized to avoid jitter) ----
            box_key = (x1 // 20, y1 // 20, x2 // 20, y2 // 20)

            # ---- CHECK CACHE ----
            if box_key in ocr_cache:
                texts = ocr_cache[box_key]
            else:
                # Preprocess
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                texts = reader.readtext(
                    gray,
                    detail=0,
                    paragraph=False
                )

                ocr_cache[box_key] = texts  # save result
                ocr_count += 1   # ⬅ increment AFTER OCR

            # ---- USE OCR RESULT ----
            detected_texts.extend(texts)

    # Annotate detected texts on the frame
    for i, text in enumerate(detected_texts):
                # Prepare text
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        # Coordinates
        x = 10
        y = 30 + i*30

        # ---- Draw red background (thick) ----
        cv2.rectangle(frame,
                    (x, y - text_h - 6),
                    (x + text_w + 8, y + 4),
                    (0, 0, 255),   # Red background
                    -1)

        # ---- Draw white text (thick) ----
        cv2.putText(frame, text, (x + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),  # White text
                    2)                # Thickness
        detected_names.append(text)

    # -----------------------------
    # Speak when detection list changes
    # -----------------------------
    #if 'last_detected_set' not in globals():
    last_detected_set = set()

    current_set = set(detected_names)

    if current_set != last_detected_set:
        for obj in current_set:
            print("Speaking:", obj)
            speak(obj)

        last_detected_set = current_set

    # Show output window
    cv2.imshow("Object Detection + TFLite Classification", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
