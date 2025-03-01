from ultralytics import YOLO

# 1. Specify the path to your trained YOLO model (best.pt, last.pt, etc.)
model_path = r"D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\models\best.pt"

# 2. Load the model
model = YOLO(model_path)

# 3. Evaluate on your dataset (test split)
# Make sure your data.yaml has "test: path/to/test/images" defined
metrics = model.val(data="D:\Canada\Subjects\Semester -1\AIDI 1003_01_CAPSTONE TERM 1\Cosmic_Navigators_Final\dataset\data.yaml", split="test")
print(metrics)


# 4. Print the results
print("Evaluation Metrics:", metrics)
