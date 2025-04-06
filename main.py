from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import nest_asyncio
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import uvicorn
import os

# Apply for async compatibility
nest_asyncio.apply()

# Initialize app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class TrafficResponse(BaseModel):
    area: str
    vehicle_count: int
    traffic_jam: bool
    road_status: str

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# NOAA satellite image fetcher
def fetch_satellite_image(area_name: str):
    image_url = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/CONUS/GEOCOLOR/1000x1000.jpg"
    print(f"Fetching NOAA image for: {area_name}")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((640, 640))
        return np.array(image)
    except Exception as e:
        print(f"Error fetching image: {e}")
        return np.ones((640, 640, 3), dtype=np.uint8)

# Detection logic
def run_yolo_detection(image_data):
    results = model(image_data)

    # Label 2 is 'car' in YOLOv5
    vehicle_count = sum([
        1 for conf, label in zip(results.xywh[0][:, 4], results.xywh[0][:, 5]) if int(label) == 2
    ])

    traffic_jam = vehicle_count >= 30

    # Deterministic road status logic
    if vehicle_count < 10:
        road_status = "clear"
    elif vehicle_count < 30:
        road_status = "busy"
    else:
        road_status = "traffic jam"

    return {
        "vehicle_count": vehicle_count,
        "traffic_jam": traffic_jam,
        "road_status": road_status
    }

# Endpoint
@app.get("/analyze", response_model=TrafficResponse)
def analyze_area(area: str = Query(..., description="Area name in Dhaka")):
    image_data = fetch_satellite_image(area)
    detection = run_yolo_detection(image_data)
    return JSONResponse(content={
        "area": area,
        "vehicle_count": detection["vehicle_count"],
        "traffic_jam": detection["traffic_jam"],
        "road_status": detection["road_status"]
    })

# Run server
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
