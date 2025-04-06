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
from datetime import datetime

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
    satellite_image_url: str

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Fetch satellite image from NASA GIBS using WMTS
def fetch_satellite_image(lat, lon):
    # Define the GIBS WMTS endpoint and parameters
    wmts_url = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{time}/250m/8/{tile_row}/{tile_col}.jpg"
    
    # Calculate tile row and column based on latitude and longitude
    tile_matrix = 8  # Zoom level
    tile_size = 180 / (2 ** tile_matrix)
    tile_row = int((90 - lat) / tile_size)
    tile_col = int((lon + 180) / tile_size)
    
    # Get the most recent date available in GIBS
    time = datetime.utcnow().strftime('%Y-%m-%d')
    
    # Construct the image URL
    image_url = wmts_url.format(time=time, tile_row=tile_row, tile_col=tile_col)
    print(f"Fetching satellite image for coordinates ({lat}, {lon}) from: {image_url}")
    
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((640, 640))  # Resize for consistency
        return np.array(image), image_url
    except Exception as e:
        print(f"Error fetching image: {e}")
        return np.ones((640, 640, 3), dtype=np.uint8), image_url

# Detection logic
def run_yolo_detection(image_data):
    results = model(image_data)
    
    # Label 2 corresponds to 'car' in YOLOv5
    vehicle_count = sum(1 for label in results.xywh[0][:, 5] if int(label) == 2)
    
    traffic_jam = vehicle_count >= 30

    # Road status logic
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

# Endpoint to analyze area based on Latitude and Longitude
@app.get("/analyze", response_model=TrafficResponse)
def analyze_area(lat: float = Query(..., description="Latitude of the area"),
                 lon: float = Query(..., description="Longitude of the area")):
    
    # Fetch satellite image based on coordinates
    image_data, image_url = fetch_satellite_image(lat, lon)
    
    # Perform detection on the fetched satellite image
    detection = run_yolo_detection(image_data)
    
    # Construct response with satellite image URL and detection details
    return JSONResponse(content={
        "area": f"Latitude: {lat}, Longitude: {lon}",
        "vehicle_count": detection["vehicle_count"],
        "traffic_jam": detection["traffic_jam"],
        "road_status": detection["road_status"],
        "satellite_image_url": image_url
    })

# Run server
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
