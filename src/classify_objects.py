import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------
# 1. Load Your YOLOv8 Model
# -------------------------------
yolo_model = YOLO(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\runs\best.pt")
print("YOLO Model Class Names:", yolo_model.names)

# -------------------------------
# 2. Load Your CNN Classification Model
# -------------------------------
cnn_model = load_model(r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\cnn\debris_classifier.h5")
class_names = ["large_debris", "medium_debris", "rocket", "satellite", "small_debris"]

# -------------------------------
# 3. Define the End-to-End Detection + Classification Function
# -------------------------------
def detect_and_classify(image_path):
    img = cv2.imread(image_path)
    results = yolo_model.predict(source=image_path, conf=0.3)
    boxes_data = results[0].boxes.data

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        # Crop the detection
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        # Classify with CNN
        crop_resized = cv2.resize(crop, (128, 128))
        crop_norm = crop_resized / 255.0
        crop_input = np.expand_dims(crop_norm, axis=0)

        prediction = cnn_model.predict(crop_input)
        refined_class = np.argmax(prediction, axis=1)[0]
        refined_label = class_names[refined_class]

        # Draw bounding box + label
        label_text = f"{refined_label} ({conf:.2f})"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label_text, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # -------------------------------
    # 4. Display the Annotated Image using Matplotlib
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detections + Classifications")
    plt.axis("off")
    plt.show()

    # Optionally, save the annotated image:
    output_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\photos_2736_jpg.rf.da632cb7162caf652be0d5233182e8af.jpg"
    cv2.imwrite(output_path, img)
    print("Annotated image saved to", output_path)

# -------------------------------
# 5. Run the Function on a Test Image
# -------------------------------
test_image = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\s8_PNG.rf.93eb87f610a8de5f84ed95122086fa28.jpg"
detect_and_classify(test_image)
