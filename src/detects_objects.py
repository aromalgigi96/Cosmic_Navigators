import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Load Your YOLOv8 Model
model_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\best.pt"
model = YOLO(model_path)

def detect_objects(image_path, conf=0.4):
    img = cv2.imread(image_path)
    results = model.predict(source=image_path, conf=conf)
    boxes_data = results[0].boxes.data

    # Draw boxes on the image
    for box in boxes_data:
        x1, y1, x2, y2, score, cls_id = box.tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Show or save the result
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detections")
    plt.show()

if __name__ == "__main__":
    # Use a raw string literal to avoid escape sequence issues
    test_image = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\0b09a4810895c687bb164354c55afbca_png_jpg.rf.8254164ca23688f86afc63e25ded4993.jpg"
    detect_objects(test_image, conf=0.4)
