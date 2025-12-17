import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr
import pyttsx3
import threading
import queue

# Load YOLO model
model = YOLO("yolov8n.pt")  # or your custom model

# Initialize EasyOCR
reader = easyocr.Reader(['en'], quantize=True)

# Open camera
cap = cv2.VideoCapture(0)

# Create window
root = tk.Tk()
root.title("Image Capture Object Detection")
root.geometry("900x700")
root.configure(bg="black")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# Video label
video_label = Label(root)
video_label.pack(pady=10)

# Status label
status_label = Label(
    root, text="Tap CAPTURE to Predict",
    font=("Arial", 20),
    fg="white", bg="black"
)
status_label.pack(pady=10)

def draw_boxes(frame, results):
    result = results[0]  # batch size = 1
    detected_names = []   # List to store detected object names
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{result.names[cls_id]} {conf:.2f}"
        
        detected_names.append(result.names[cls_id])  # store name
        
        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (int(x1), int(y1 - th - 5)), (int(x1 + tw), int(y1)), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, (int(x1), int(y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame, detected_names

def speak_text(text):
    """Create a fresh engine each time to allow multiple speeches."""
    engine = pyttsx3.init()  # new engine for each call
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Capture & predict function
def capture_predict():
    status_label.config(text="Predicting...")
    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Camera Error")
        return

    # --- YOLO Object Detection ---
    results = model(frame)
    frame, detected_objects = draw_boxes(frame, results)
   
    # --- EasyOCR Text Detection ---
    detected_texts = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            roi = frame[y1:y2, x1:x2]
            text_results = reader.readtext(roi)

            for (_, text, conf) in text_results:
                if conf > 0.5:
                    detected_texts.append(text)

    # --- Show annotated frame ---
    show_frame(frame)

    # --- Update GUI status label ---
    status_text = ""
    if detected_objects:
        status_text += "Detected Objects: " + ", ".join(detected_objects)
    if detected_texts:
        if status_text:
            status_text += "\n"
        status_text += "Detected Texts: " + ", ".join(detected_texts)
    if not status_text:
        status_text = "No objects or text detected"

    status_label.config(text=status_text)
    #speech_queue.put("Hello, you clicked the button!")
    threading.Thread(target=speak_text, args=(status_text,), daemon=True).start()


# Show frame in GUI
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1150, 800))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

# Update live video
def update_video():
    ret, frame = cap.read()
    if ret:
        show_frame(frame)
    root.after(10, update_video)

# Large touch-friendly button
capture_btn = Button(
    root,
    text="CAPTURE & PREDICT",
    command=capture_predict,
    font=("Arial", 24, "bold"),
    bg="green",
    fg="white",
    width=20,
    height=2
)
capture_btn.pack(pady=20)

# Start video loop
update_video()

# Clean exit
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
