from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from ai_engine import AITracker

app = FastAPI(title="AI Behavioral Analysis API")
tracker = AITracker()

class CalibrateRequest(BaseModel):
    mode: str  # 'Happy', 'Sad', 'Angry', 'Natural'

class FrameRequest(BaseModel):
    image: str  # Base64 encoded image string

@app.get("/")
def read_root():
    return {"message": "AI Tracker API is running (Session Mode)"}

@app.get("/state")
def get_state():
    """Returns the latest emotion and distraction state"""
    return tracker.get_latest_state()

@app.post("/analyze_frame")
def analyze_frame(req: FrameRequest):
    """Process a single base64 image frame"""
    result = tracker.process_frame_base64(req.image)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result

@app.post("/calibrate")
def calibrate_emotion(req: CalibrateRequest):
    """Start a 5-second calibration for the specified mode"""
    valid_modes = ['Happy', 'Sad', 'Angry', 'Natural']
    if req.mode not in valid_modes:
        return JSONResponse(status_code=400, content={"error": "Invalid mode. Use Happy, Sad, Angry, or Natural"})
    
    tracker.set_calibration_mode(req.mode)
    return {"message": f"Calibration for {req.mode} started. Please hold the expression for 5 seconds."}

@app.post("/start_session")
def start_session():
    """Starts tracking statistics for a new session"""
    return tracker.start_session()

@app.post("/end_session")
def end_session():
    """Ends the current session and returns the final calculated statistics"""
    return tracker.end_session()

if __name__ == "__main__":
    try:
        print("[INFO] Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        tracker.stop()
