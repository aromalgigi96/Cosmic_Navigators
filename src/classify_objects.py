import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Your YOLOv8 Model
# -------------------------------
model_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\best.pt"
model = YOLO(model_path)

# Print the model's class names to verify (e.g., {0: 'large_debris', 1: 'medium_debris', ...})
print("Model Class Names:", model.names)

# -------------------------------
# 2. Run Inference on an Image
# -------------------------------
image_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\2.jpg"

# Run inference with a confidence threshold of 0.5
results = model.predict(source=image_path, conf=0.4)

# Print the overall results object
print("Raw Results Object:", results)

# Extract raw bounding box data from the first image result.
boxes_data = results[0].boxes.data  # Tensor of shape (N, 6): [x1, y1, x2, y2, conf, class_id]
print("\nRaw Boxes Tensor:\n", boxes_data)

# -------------------------------
# 3. Parse and Print Each Detection
# -------------------------------
print("\nParsed Detections:")
for i, box in enumerate(boxes_data):
    x1, y1, x2, y2, conf, cls_id = box.tolist()  # Convert tensor to Python floats
    class_name = model.names.get(int(cls_id), "unknown")
    print(f"Detection {i+1}: [x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, conf={conf:.2f}, class={class_name}]")

# -------------------------------
# 4. Visualize the Detections
# -------------------------------
# Read the image using OpenCV
img = cv2.imread(image_path)

# Draw bounding boxes and labels on the image
for box in boxes_data:
    x1, y1, x2, y2, conf, cls_id = box.tolist()
    class_name = model.names.get(int(cls_id), "unknown")
    # Draw rectangle: top-left (x1,y1) and bottom-right (x2,y2)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Put text above the bounding box
    label = f"{class_name} {conf:.2f}"
    cv2.putText(img, label, (int(x1), int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detections")
plt.axis("off")
plt.show()

# -------------------------------
# 5. Save the Annotated Image
# -------------------------------
output_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\detections_result.jpg"
cv2.imwrite(output_path, img)
print(f"Saved detection result to {output_path}")
