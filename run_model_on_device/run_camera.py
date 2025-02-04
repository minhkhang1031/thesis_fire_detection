import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import yaml
import os
import time
import subprocess
import threading

model_folder = "best_ncnn_model"
yaml_path = os.path.join(model_folder, "metadata.yaml")

ncnn_model = YOLO(model_folder)

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

class_names = load_class_names(yaml_path)

threshold = 0.3
alert_fire = ["fire", "smoke"]
alert_lock = threading.Lock()

def play_alert_sound():
    with alert_lock:
        for _ in range(1):
            subprocess.call(['mpg321', "alert_sound.mp3"])
            time.sleep(10)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = ncnn_model(frame_bgr)
        fire_detected = False

        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls

            for i, box in enumerate(boxes):
                confidence = confidences[i]
                class_id = int(class_ids[i])
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

                if confidence >= threshold and class_name in alert_fire:
                    fire_detected = True

        if fire_detected:
            threading.Thread(target=play_alert_sound, daemon=True).start()

        time.sleep(0.1)
finally:
    picam2.stop()
