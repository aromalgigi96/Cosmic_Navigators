import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Class names for detection & classification
class_names = ["large_debris", "medium_debris", "rocket", "satellite", "small_debris"]

# Predefined colors for each class (BGR format)
# Feel free to adjust or add more colors
CLASS_COLORS = {
    "large_debris":  (0, 255, 0),    # green
    "medium_debris": (255, 0, 0),    # blue
    "rocket":        (0, 0, 255),    # red
    "satellite":     (255, 255, 0),  # cyan
    "small_debris":  (255, 0, 255)   # magenta
}

def draw_label_with_bg(img, text, x, y, color, font_scale=0.5):
    """
    Draws a label with a semi-transparent background for better visibility.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Add some padding
    pad = 3
    text_x2 = x + text_w + 2*pad
    text_y2 = y - text_h - 2*pad

    # Draw a filled rectangle (semi-transparent) behind the text
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (text_x2, text_y2), color, -1)
    # Add transparency
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw the text on top
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_image_with_style(image_path, detection_model, classification_model,
                             expected_detection_count=3, detection_count_tolerance=1):
    # Load image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("Error: Unable to load image!")
        return
    annotated_img = orig_img.copy()

    # YOLO detection
    results = detection_model.predict(source=image_path, conf=0.3, iou=0.45, max_det=1000)
    boxes = results[0].boxes.data  # (N, 6): [x1, y1, x2, y2, conf, cls]

    detection_count = boxes.shape[0]

    # Simple anomaly check
    anomaly_flag = False
    anomaly_text = ""
    if detection_count < (expected_detection_count - detection_count_tolerance):
        anomaly_flag = True
        anomaly_text = f"Anomaly: Too few objects ({detection_count})"
    elif detection_count > (expected_detection_count + detection_count_tolerance):
        anomaly_flag = True
        anomaly_text = f"Anomaly: Too many objects ({detection_count})"

    # If anomaly, show text in red, else green
    anomaly_color = (0, 0, 255) if anomaly_flag else (0, 255, 0)

    # Put anomaly message at the top
    draw_label_with_bg(annotated_img, anomaly_text if anomaly_text else f"Detections: {detection_count}",
                       10, 30, anomaly_color, font_scale=0.7)

    # For each detection, do classification & draw bounding box
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf_str = f"{conf:.2f}"

        # Crop region
        crop = orig_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Classification with your CNN
        crop_resized = cv2.resize(crop, (128, 128))
        crop_norm = crop_resized / 255.0
        crop_input = np.expand_dims(crop_norm, axis=0)
        pred = classification_model.predict(crop_input)
        refined_class_id = np.argmax(pred, axis=1)[0]
        refined_class_name = class_names[refined_class_id]

        # YOLO's label
        detect_class_name = class_names[int(cls_id)]
        final_label = f"{detect_class_name}/{refined_class_name} {conf_str}"

        # Determine color
        # We'll pick the color for the bounding box based on the detection class
        color = CLASS_COLORS.get(detect_class_name, (0, 255, 0))  # default green if not found

        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        # Draw label above bounding box with background
        draw_label_with_bg(annotated_img, final_label, x1, y1 - 5, color)

    # Show final image with matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Detection, Classification & Anomaly Check (Stylized)")
    plt.axis("off")
    plt.show()

    # Optionally save the result
    output_path = "styled_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"Result saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Load detection model
    yolo_model = YOLO(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\runs\best.pt")

    # Load classification model
    from tensorflow.keras.models import load_model
    cnn_model = load_model(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\cnn\debris_classifier.h5")

    # Process an example image
    image_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\0fd573764fa41dd591236d879eae0280_png_jpg.rf.0fe33c8b4a9c57babc807d17d387cd04.jpg"
    process_image_with_style(image_path, yolo_model, cnn_model,
                             expected_detection_count=3, detection_count_tolerance=2)
