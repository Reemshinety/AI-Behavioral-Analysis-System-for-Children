# Project Documentation
## AI Behavioral Analysis System for Children

---

### Abstract
This document covers the final phase of the graduation project, focusing on the complete implementation of the AI Behavioral Analysis System. It details the transition from initial prototypes to a fully integrated, API-driven Microservice architecture. The project encompasses rigorous performance and accuracy testing of the facial expression and gaze-tracking models, alongside the development of a comprehensive user manual. The final system seamlessly integrates an AI Engine (Python/MediaPipe) with a mobile frontend (Flutter) and a backend server (Node.js).

---

### Chapter 1: Introduction
This chapter provides a general overview of the project and the achievements completed during the first phase. The primary goal of this final phase is the complete execution and integration of the system components. 

In the previous phase, the core concept of utilizing AI to monitor children's attention and emotions was established. The objectives for this final phase included transitioning the AI model from a local, webcam-dependent script into a robust, stateless RESTful API. This allows the system to be deployed on cloud servers and communicate in real-time with a Flutter mobile application, ensuring a seamless tracking lifecycle during educational gameplay without disrupting the user experience.

---

### Chapter 2: Implementation
This chapter details the software implementation process, the development environment, tools, programming languages used, and a description of the main software modules.

#### 2.1 Technologies and Tools
*   **AI & Computer Vision:** Python, OpenCV, Google MediaPipe (Tasks Vision API).
*   **Backend & API Services:** FastAPI (Python) for the AI Microservice.
*   **Architecture:** Microservices Architecture (Stateless API-driven frame processing).

#### 2.2 System Modules Description
1.  **AI Tracking Engine (`ai_engine.py`):** The core module responsible for extracting facial landmarks, calculating head pose (Pitch, Yaw, Roll) using a 3D face model, and computing gaze ratios (iris tracking). It features an advanced calibration system that adjusts to the user's natural resting face to prevent false positives.
2.  **FastAPI Server (`server.py`):** A lightweight web server that exposes the AI Engine to the outside world. It includes endpoints like `POST /analyze_frame` to receive Base64 encoded images from the mobile app and return instantaneous JSON analysis.

#### 2.3 System Integration
The partial models were integrated by decoupling the camera capture logic from the AI processing logic. Instead of the Python script accessing the physical camera, the Flutter app captures frames periodically (e.g., 1 frame per second) and sends them to the Python API. The Python API analyzes the frame, updates the active session statistics, and upon session completion, the Flutter app retrieves the final behavioral report and sends it to the Node.js database.

---

### Chapter 3: Testing and Experiments
This chapter presents the testing plan, test cases applied to the system, and the results related to performance, accuracy, and reliability.

#### 3.1 Accuracy Testing (Emotion and Gaze)
*   **Test Case:** Child looking away from the screen (head movement vs. eye movement).
*   **Result:** The system accurately detects distraction using strict thresholds (10 degrees for head pitch/yaw, 0.08 ratio for iris movement).
*   **Test Case:** Differentiating between natural expressions and sadness.
*   **Result:** Initially, looking up caused false "Sadness" triggers (due to inner brow raising). An algorithmic penalty was introduced using the `eyeLookUp` blendshape to successfully filter out these false positives.

#### 3.2 Performance and Reliability
*   **Test Case:** API Latency under continuous frame streaming.
*   **Result:** By converting the system to process Base64 images via REST API, the response time per frame was reduced to milliseconds. The stateless design ensures no memory leaks occur during long gameplay sessions.

---

### Chapter 4: Results
This chapter presents the final results of the system, including system screens, performance statistics, situation detection, and a comparison with similar solutions.

#### 4.1 Final Outputs
*   **Real-time Detection:** The system successfully identifies states such as "Focused" vs. "Distracted (Head Down, Eyes Left)".
*   **Session Reports:** At the end of a session, a structured JSON report is generated containing the exact duration of the session, the total number of frames analyzed, the overall distraction percentage, and a breakdown of emotional distribution (Happy, Sad, Angry, Fearful, Natural).
*   **Dashboard Visualization:** The Node.js backend saves these reports, allowing parents to view historical data of their child's attention span and emotional well-being over time.

#### 4.2 Comparison with Existing Solutions
Unlike traditional eye-tracking hardware which is expensive and requires specific hardware, this system operates entirely via standard webcams and smartphone cameras. Furthermore, unlike generic emotion APIs, this system utilizes a *Personal Calibration Module*, learning the child's specific baseline face to significantly increase accuracy for neurodivergent children.

---

### Chapter 5: Discussion
This chapter analyzes the results, the extent to which the system achieved its goals, the challenges faced by the team, and how they were handled.

#### 5.1 Achieving Goals
The primary goal of creating a non-intrusive, highly accurate behavioral tracking system was achieved. The Microservice architecture ensures high scalability.

#### 5.2 Challenges and Solutions
*   **Challenge 1: Server Camera Dependency:** The initial Python prototype relied on `cv2.VideoCapture(0)`, tying the AI to the local machine's hardware.
    *   *Solution:* The architecture was entirely refactored to an API-driven model. The `ai_engine.py` was rewritten to accept Base64 image frames over HTTP, separating the camera logic from the AI logic.
*   **Challenge 2: Network Latency & Bandwidth:** Sending 30 frames per second from a mobile device to a server would crash the network.
    *   *Solution:* Implemented a sampling strategy in Flutter, capturing and sending only 1 frame every second (or an interval determined by the timer), drastically reducing bandwidth while maintaining high analytical accuracy.

---

### Chapter 6: Conclusion and Recommendations
This chapter summarizes the project's main achievements and provides recommendations for future development.

#### 6.1 Conclusion
The AI Behavioral Analysis System provides a robust, scalable, and privacy-conscious method for tracking a child's attention and emotional state during educational activities. The separation of concerns between the Flutter frontend, Python AI engine, and Node.js database represents a professional-grade software architecture.

#### 6.2 Future Recommendations
*   **Edge Computing (On-Device Processing):** Migrating the MediaPipe Python logic directly into the Flutter app using native plugins to eliminate network latency entirely and ensure 100% offline functionality.
*   **Advanced Analytics:** Integrating machine learning models on the Node.js backend to predict when a child is about to lose interest based on historical session data.
*   **Parental Alerts:** Implementing real-time push notifications to parents if severe distress or prolonged distraction is detected.

---

### References
1.  Google MediaPipe Documentation: Face Landmarker and Blendshapes.
2.  OpenCV (Open Source Computer Vision Library) Documentation.
3.  FastAPI Framework Documentation.
4.  Flutter Camera Plugin and HTTP Networking Guidelines.

---

### Appendices

#### A. User Manual (Developer & Setup Guide)
1.  **Running the AI Server:**
    *   Navigate to the Python directory.
    *   Install requirements: `pip install -r requirements.txt`
    *   Start server: `python server.py`
    *   The API will be available at `http://localhost:8000`.
2.  **Testing the API Locally:**
    *   Run `python test_client.py` to capture a frame from the local webcam and test the `/analyze_frame` endpoint.
3.  **Flutter Integration:**
    *   Ensure the Flutter app's `baseUrl` in `ai_service.dart` points to the correct server IP.

#### B. Final Code Repository Links
*   *(Insert actual GitHub/GitLab links to the Python AI Engine, Node.js Backend, and Flutter App here)*

#### C. System Screenshots
*   *(Attach screenshots of the Flutter game interface, the parent dashboard showing session reports, and the API JSON responses here)*

#### D. Explanatory Video
*   *(Provide a link to a YouTube or Google Drive video demonstrating the real-time tracking and the final session report generation)*
