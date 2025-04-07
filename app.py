from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import tempfile
import os
import uuid
import io

# Import YOLO and load TensorFlow model
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Import PDF generation function from pdf.py
from pdf import generate_pdf

# Update model paths
yolo_model = YOLO("./models/yolo/best.pt")
resnet_model = load_model("./models/cnn/debris_resnet_classifier.h5")

# Class names for detection
class_names = ["large_debris", "medium_debris", "rocket", "satellite"]

app = FastAPI()

# Allow CORS (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dict to store uploaded video paths keyed by video_id
uploaded_videos = {}

def detect_and_get_details(frame_bgr, conf_threshold=0.3):
    """
    Process a single frame:
      - Perform YOLO detection.
      - Classify each detected region with ResNet.
      - Draw bounding boxes and labels on the frame.
      - Return the annotated frame and a list of detection details (label, coordinates, and dimensions).
    """
    results = yolo_model.predict(source=frame_bgr, conf=conf_threshold)
    boxes_data = results[0].boxes.data  # [x1, y1, x2, y2, conf, class_id]
    detection_details = []
    annotated = frame_bgr.copy()

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = class_names[int(cls_id)]
        width = x2 - x1
        height = y2 - y1
        detection_details.append({
            "label": label,
            "coordinates": (x1, y1, x2, y2),
            "dimensions": (width, height)
        })
        # Draw bounding box & label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return annotated, detection_details

@app.get("/")
def root():
    return {"message": "Welcome to Cosmic Navigator API"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    pdf: bool = Query(False, description="Set to true to generate a PDF report for images"),
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Detection confidence threshold")
):
    """
    For images:
      - Returns annotated JPEG unless ?pdf=true is provided, in which case a PDF report is returned.
    For videos:
      - Saves the video temporarily and returns a video_id.
    """
    filename = file.filename.lower()
    contents = await file.read()

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        annotated_frame, details = detect_and_get_details(frame, conf_threshold=threshold)
        if pdf:
            try:
                pdf_buffer = generate_pdf(annotated_frame, details)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return StreamingResponse(
                pdf_buffer, 
                media_type="application/pdf", 
                headers={"Content-Disposition": "attachment; filename=result.pdf"}
            )
        else:
            ret, buf = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                raise HTTPException(status_code=500, detail="Failed to encode image.")
            return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    elif filename.endswith(('.mp4', '.avi', '.mov')):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            video_path = tmp.name
        video_id = str(uuid.uuid4())
        uploaded_videos[video_id] = video_path
        return JSONResponse({"message": "Video uploaded", "video_id": video_id})
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

@app.get("/stream_video/{video_id}")
def stream_video(
    video_id: str,
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Detection confidence threshold for video streaming")
):
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found.")
    video_path = uploaded_videos[video_id]
    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video.")
        skip_frames = 2
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            annotated_frame, _ = detect_and_get_details(frame, conf_threshold=threshold)
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        cap.release()
        os.remove(video_path)
        del uploaded_videos[video_id]
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
