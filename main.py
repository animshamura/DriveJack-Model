from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import torch
import numpy as np
import random
import uvicorn
from pydantic import BaseModel
from fastapi import Query
from fastapi.responses import JSONResponse
import os

# Apply nest_asyncio
nest_asyncio.apply()

# FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Response model
class TrafficResponse(BaseModel):
    area: str
    vehicle_count: int
    traffic_jam: bool
    road_status: str

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Simulate satellite image fetching
def fetch_satellite_image(area_name: str):
    print(f"Fetching satellite image for: {area_name}")
    return f"mocked_image_data_for_{area_name}"

# Simulated image processing
def process_image(image_data):
    image = np.ones((640, 640, 3), dtype=np.uint8)  # Dummy white image
    return image

# YOLO detection
def run_yolo_detection(image_data):
    image = process_image(image_data)
    results = model(image)
    vehicle_count = sum([1 for conf, label in zip(results.xywh[0][:, 4], results.xywh[0][:, 5]) if label == 2])
    jam = vehicle_count > 30
    road_status = random.choice(["clear", "blocked", "construction"])
    return {
        "vehicle_count": vehicle_count,
        "traffic_jam": jam,
        "road_status": road_status
    }

# API endpoint
@app.get("/analyze", response_model=TrafficResponse)
def analyze_area(area: str = Query(..., description="Area name in Dhaka")):
    image_data = fetch_satellite_image(area)
    detection = run_yolo_detection(image_data)
    data = {
        "area": area,
        "vehicle_count": detection["vehicle_count"],
        "traffic_jam": detection["traffic_jam"],
        "road_status": detection["road_status"]
    }
    return JSONResponse(content=data)

# Run app with Uvicorn (or any other ASGI server like Gunicorn in production)
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")  # Host is 0.0.0.0 for any IP
    port = int(os.getenv("PORT", 8000))  # Default port is 8000, can be set via env variables
    uvicorn.run(app, host=host, port=port)
