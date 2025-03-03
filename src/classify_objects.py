import os
# Workaround for the "multiple copies of libiomp5md.dll" OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load your models
yolo_model = YOLO("models/best.pt")
cnn_model = load_model("models/cnn/debris_classifier.h5")

class_names = {
    0: 'large_debris',
    1: 'medium_debris',
    2: 'rocket',
    3: 'satellite',
    4: 'small_debris'
}

def detect_and_classify(image_path, conf=0.4):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # --- DETECTION (YOLO) ---
    results = yolo_model.predict(source=image_path, conf=conf)
    boxes_data = results[0].boxes.data  # [x1, y1, x2, y2, confidence, class_id]

    # --- CLASSIFICATION (CNN) ---
    for box in boxes_data:
        x1, y1, x2, y2, score, cls_id = box.tolist()

        # Crop the region
        cropped = img[int(y1):int(y2), int(x1):int(x2)]
        if cropped.size == 0:
            # If the bounding box is invalid (e.g., out of image bounds), skip it
            continue

        # Preprocess for CNN
        cropped_resized = cv2.resize(cropped, (128, 128))
        cropped_norm = cropped_resized / 255.0
        cropped_input = np.expand_dims(cropped_norm, axis=0)

        # Predict refined class
        prediction = cnn_model.predict(cropped_input)
        refined_class = np.argmax(prediction, axis=1)[0]

        label = f"{class_names[refined_class]} {score:.2f}"

        # Draw bounding box & label
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the annotated image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detections + Classifications")
    plt.show()

if __name__ == "__main__":
    # Use a raw string (r"") for Windows paths to avoid escape sequence issues
    test_image = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\0b09a4810895c687bb164354c55afbca_png_jpg.rf.8254164ca23688f86afc63e25ded4993.jpg"
    detect_and_classify(test_image)
