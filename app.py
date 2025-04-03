from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Update paths 
yolo_model = YOLO("./models/yolo/best.pt")
resnet_model = load_model("./models/cnn/debris_resnet_classifier.h5")

class_names = ["large_debris", "medium_debris", "rocket", "satellite"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dict to store uploaded video paths keyed by video_id
uploaded_videos = {}

def detect_and_classify_frame(frame_bgr, conf_threshold=0.3):
    """
    Process a single frame:
      - YOLO detection.
      - Crop and classify each detected object with ResNet.
      - Draw bounding boxes and labels on the frame.
    """
    results = yolo_model.predict(source=frame_bgr, conf=conf_threshold)
    boxes_data = results[0].boxes.data  
    annotated = frame_bgr.copy()

    for box in boxes_data:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = annotated[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Prepare the crop for ResNet
        crop_resized = cv2.resize(crop, (128, 128))
        crop_norm = crop_resized / 255.0
        crop_input = np.expand_dims(crop_norm, axis=0)

        prediction = resnet_model.predict(crop_input)
        refined_class_id = int(np.argmax(prediction, axis=1)[0])
        label = class_names[refined_class_id]

        # Draw bounding box & label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} ({conf:.2f})",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return annotated

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles both images and videos:
      - If it's an image, return the annotated image immediately as a binary stream.
      - If it's a video, save it temporarily and return JSON with a 'video_id'.
    """
    filename = file.filename.lower()
    contents = await file.read()

    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Handle image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        annotated_frame = detect_and_classify_frame(frame)
        ret, buf = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode image.")
        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/jpeg"
        )

    elif filename.endswith(('.mp4', '.avi', '.mov')):
        # Handle video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            video_path = tmp.name

        video_id = str(uuid.uuid4())
        uploaded_videos[video_id] = video_path

        return JSONResponse({"message": "Video uploaded", "video_id": video_id})
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

@app.get("/stream_video/{video_id}")
def stream_video(video_id: str):
    """
    Streams an MJPEG video of annotated frames for the given video_id.
    The React frontend sets <img src=".../stream_video/{video_id}" /> for videos.
    """
    if video_id not in uploaded_videos:
        raise HTTPException(status_code=404, detail="Video not found.")
    video_path = uploaded_videos[video_id]

    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video.")

        skip_frames = 2  # Process every 2nd frame to speed up streaming
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            annotated_frame = detect_and_classify_frame(frame)
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        cap.release()
        # Cleanup: remove file & dictionary entry
        os.remove(video_path)
        del uploaded_videos[video_id]

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
