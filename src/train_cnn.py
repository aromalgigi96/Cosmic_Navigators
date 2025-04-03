import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Class names for detection & classification
class_names = ["large_debris", "medium_debris", "rocket", "satellite", "small_debris"]

# Predefined colors for each class (BGR format)
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
    pad = 3  # extra padding around text

    # Coordinates for the background rectangle
    text_x2 = x + text_w + 2 * pad
    text_y2 = y - text_h - 2 * pad

    # Draw a filled rectangle (semi-transparent) behind the text
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (text_x2, text_y2), color, -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw the text on top (white text)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_image_show_count_and_labels(image_path, detection_model, classification_model):
    # Load image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("Error: Unable to load image!")
        return
    annotated_img = orig_img.copy()

    # Run YOLO detection
    results = detection_model.predict(source=image_path, conf=0.3, iou=0.45, max_det=1000)
    boxes = results[0].boxes.data  # shape: (N, 6) => [x1, y1, x2, y2, conf, cls]

    # Count how many detections
    detection_count = boxes.shape[0]

    # Show the number of detections at the top-left corner
    draw_label_with_bg(annotated_img,
                       f"Detections: {detection_count}",
                       10, 30,
                       (0, 255, 0),  # green background
                       font_scale=0.7)

    # For each detection, refine the class with your CNN and draw a bounding box + label
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf_str = f"{conf:.2f}"

        # Crop region from the original image
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

        # Determine color based on YOLO’s detection class
        detect_class_name = class_names[int(cls_id)]
        color = CLASS_COLORS.get(detect_class_name, (0, 255, 0))

        # Create label: just show the refined class + YOLO confidence
        final_label = f"{refined_class_name} ({conf_str})"

        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        # Draw label above bounding box
        draw_label_with_bg(annotated_img, final_label, x1, y1 - 5, color)

    # Show final image with matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Detection Count + Refined Labels")
    plt.axis("off")
    plt.show()

    # Optionally save the result
    output_path = "count_and_labels_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"Result saved to {output_path}")

# -------------------
# Example usage:
# -------------------
if __name__ == "__main__":
    # 1. Load YOLO detection model
    yolo_model = YOLO(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\runs\best.pt")

    # 2. Load CNN classification model
    cnn_model =  load_model(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\cnn\debris_resnet_classifier.h5")

    # 3. Input image path
    image_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\-30_png_jpg.rf.6947af000ca40f01104b1eba7f7a9a59.jpg"

    # Process the image
    process_image_show_count_and_labels(image_path, yolo_model, cnn_model)
