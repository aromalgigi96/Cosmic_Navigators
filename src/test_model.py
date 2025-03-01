import torch
from ultralytics import YOLO

# Load the model
model_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\best.pt"
model = YOLO(model_path)

# Path to your input image
img_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\test\images\-31_png.rf.357ff125e6b4bec4bd1b451b3e62d593.jpg"

# Perform inference with a lower confidence threshold (e.g., 0.20) to see more bounding boxes

results = model.predict(source=img_path, conf=0.04, iou=0.45, max_det=1000)

# Access first result
result = results[0]

# Print all bounding boxes in raw format
print(result.boxes)          # Summarized boxes
print(result.boxes.xyxy)     # Coordinates + confidence + class
print(result.boxes.conf)     # Confidence scores
print(result.boxes.cls)      # Class IDs


# Access the first result if multiple results exist
result = results[0]

# Print detection details (bounding boxes, class IDs, etc.)
print(result.verbose())

# Display the result image with bounding boxes
result.show()

# Save the result image with bounding boxes in the 'runs' directory
result.save()
