🌌 Cosmic Navigators – AI-Powered Space Debris Detection System
    A capstone project aimed at building a lightweight, end-to-end system for detecting, classifying, and tracking space debris from user-uploaded images and videos using cutting-edge AI and cloud technologies.

🚀 Project Overview

    Problem:

        Space debris poses a growing risk to satellites and spacecraft, yet monitoring tools are often inaccessible to non-experts.

    Our Solution:

        A full-stack, web-based platform that leverages computer vision to detect and classify space debris in real-time.

🛠️ Tech Stack

        Layer	Tools/Technologies
        Frontend	React.js (hosted on AWS S3)
        Backend	FastAPI (Dockerized, hosted on AWS EC2)
        AI Models	YOLOv8 (detection), ResNet50 (classification)
        Deployment	Docker, AWS EC2, AWS ECR, AWS S3
🎯 Features

        Upload images or videos to detect space debris

        Real-time bounding box predictions and classification

        Adjustable confidence threshold for results

        Export results as annotated images or PDF reports

        Stream video with live annotation

        Bonus: Space-themed chatbot for space data

🧠 Model Architecture

        YOLOv8: For real-time object detection (bounding boxes)

        ResNet50: Transfer learning for classifying debris types

        Custom Tracker: Logic-based tracking using bounding box coordinates

🔢 Evaluation Metrics

        Model	Metric	Result
        YOLOv8	mAP	~88%
        ResNet50	Accuracy	~90%
        Inference Time	Avg/image	~0.8 seconds
📦 Setup Instructions

Clone the Repository

        bash
        Copy
        Edit
        git clone [REPO_URL]
        cd cosmic-navigators
        Run Docker Backend

        bash
        Copy
        Edit
        docker build -t debris-api .
        docker run -p 8000:8000 debris-api
        Run Frontend

        Deploy React app to S3 static hosting

        Configure API endpoints in .env



📂 Repository Contents

        backend/: FastAPI app, YOLO & ResNet models

        frontend/: React app interface

        models/: Trained model weights

        deployment/: Dockerfiles, AWS setup

        notebooks/: Training and evaluation scripts

🌍 Reproducibility

        Clone the repo and follow setup steps

        Models load automatically from /models

        Supports API requests via Postman or UI

