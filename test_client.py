import requests
import base64
import time
import cv2

def capture_test_frame():
    print("Capturing frame from local camera for testing...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to grab frame from webcam")
        return None
    return frame

def test_api():
    frame = capture_test_frame()
    if frame is None:
        return
        
    # Encode as JPEG and then Base64
    _, buffer = cv2.imencode('.jpg', frame)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    
    url = "http://127.0.0.1:8000/analyze_frame"
    payload = {"image": b64_string}
    
    print(f"Sending frame to {url}...")
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        print(f"Request took: {end_time - start_time:.3f} seconds")
        print("Status Code:", response.status_code)
        import json
        print("Response JSON:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error connecting to server:", e)
        print("Make sure server.py is running!")

if __name__ == "__main__":
    test_api()
