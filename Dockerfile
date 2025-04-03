# Use a slim base image with Debian Bullseye (minimal OS)
FROM python:3.9-slim-bullseye

# Install necessary system dependencies for OpenCV and cleaning up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies without caching wheels
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (app.py, models, etc.)
COPY . .

# Expose the port that uvicorn will run on
EXPOSE 8000

# Start your FastAPI application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
