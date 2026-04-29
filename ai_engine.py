import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
import base64

class AITracker:
    def __init__(self):
        # MediaPipe Initialization
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # 3D Model for Head Pose
        self.face_3d_model = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # State & Calibration
        self.personal_profiles = {'Happy': None, 'Sad': None, 'Angry': None, 'Natural': None}
        self.center_baseline = {'pitch': 0.0, 'yaw': 0.0, 'horiz': 0.5, 'vert': 0.5}
        self.is_center_calibrated = False

        self.calibration_mode = None
        self.calibration_start_time = 0
        self.calibration_duration = 5.0
        self.calibration_data_buffer = []
        self.calibration_distraction_buffer = []
        
        # Auto-Calibration
        self.auto_center_buffer = []
        self.auto_calibrate_frames = 10  # Reduced for API usage where frame rate is lower

        # Session Management
        self.session_active = False
        self.session_start_time = 0
        self.session_end_time = 0
        self.stats = {
            "total_samples": 0,
            "distracted_samples": 0,
            "emotions": {
                "Happy": 0,
                "Sad": 0,
                "Angry": 0,
                "Natural (Neutral)": 0,
                "Fearful": 0
            }
        }
        
        # Processing control
        self.last_process_time = 0

        # State Protection
        self.lock = threading.Lock()
        self.latest_state = {
            "emotion": "Natural (Neutral)",
            "status": "Waiting for frames...",
            "emotion_scores": {},
            "distraction_reasons": []
        }

    def stop(self):
        pass # No background thread to stop anymore

    def set_calibration_mode(self, mode):
        with self.lock:
            self.calibration_mode = mode
            self.calibration_start_time = time.time()
            self.calibration_data_buffer = []
            if mode == 'Natural':
                self.calibration_distraction_buffer = []

    def start_session(self):
        with self.lock:
            self.session_active = True
            self.session_start_time = time.time()
            self.stats = {
                "total_samples": 0,
                "distracted_samples": 0,
                "emotions": {
                    "Happy": 0,
                    "Sad": 0,
                    "Angry": 0,
                    "Natural (Neutral)": 0,
                    "Fearful": 0
                }
            }
            return {"message": "Session started", "start_time": self.session_start_time}

    def end_session(self):
        with self.lock:
            self.session_active = False
            self.session_end_time = time.time()
            duration = self.session_end_time - self.session_start_time
            
            total = self.stats["total_samples"]
            report = {
                "start_time": self.session_start_time,
                "end_time": self.session_end_time,
                "duration_seconds": round(duration, 2),
                "total_samples": total,
                "distraction_percentage": 0.0,
                "emotion_percentages": {}
            }
            
            if total > 0:
                report["distraction_percentage"] = round((self.stats["distracted_samples"] / total) * 100, 2)
                for em, count in self.stats["emotions"].items():
                    report["emotion_percentages"][em] = round((count / total) * 100, 2)
                    
            return report

    def get_latest_state(self):
        with self.lock:
            state = self.latest_state.copy()
            state["session_active"] = self.session_active
            if self.session_active:
                state["session_duration"] = round(time.time() - self.session_start_time, 2)
            return state

    # --- Core Logic Functions ---
    def _get_blendshape(self, blendshapes, name):
        for b in blendshapes:
            if b.category_name == name: return b.score
        return 0.0

    def _get_raw_emotion_scores(self, blendshapes):
        smile_l = self._get_blendshape(blendshapes, 'mouthSmileLeft')
        smile_r = self._get_blendshape(blendshapes, 'mouthSmileRight')
        brow_down_l = self._get_blendshape(blendshapes, 'browDownLeft')
        brow_down_r = self._get_blendshape(blendshapes, 'browDownRight')
        brow_inner_up = self._get_blendshape(blendshapes, 'browInnerUp')
        frown_l = self._get_blendshape(blendshapes, 'mouthFrownLeft')
        frown_r = self._get_blendshape(blendshapes, 'mouthFrownRight')
        eye_wide_l = self._get_blendshape(blendshapes, 'eyeWideLeft')
        eye_wide_r = self._get_blendshape(blendshapes, 'eyeWideRight')
        jaw_open = self._get_blendshape(blendshapes, 'jawOpen')

        happy_score = (smile_l + smile_r) / 2.0
        angry_score = (brow_down_l + brow_down_r) / 2.0
        frown_avg = (frown_l + frown_r) / 2.0
        sad_score = (brow_inner_up + frown_avg) / 2.0

        eye_look_up = (self._get_blendshape(blendshapes, 'eyeLookUpLeft') + self._get_blendshape(blendshapes, 'eyeLookUpRight')) / 2.0
        if eye_look_up > 0.2:
            sad_score -= (eye_look_up * 0.5)

        fear_score = (eye_wide_l + eye_wide_r + jaw_open) / 3.0

        return {
            "Happy": max(0.0, happy_score),
            "Angry": max(0.0, angry_score),
            "Sad": max(0.0, sad_score),
            "Fearful": max(0.0, fear_score)
        }

    def _detect_emotion_calibrated(self, raw_emotions):
        adjusted_emotions = {}
        natural_baseline = self.personal_profiles.get('Natural')
        
        if natural_baseline:
            for em in raw_emotions:
                adjusted_val = raw_emotions[em] - natural_baseline.get(em, 0.0)
                adjusted_emotions[em] = max(0.0, adjusted_val)
        else:
            adjusted_emotions = dict(raw_emotions)
            
        relative_scores = {}
        for em in adjusted_emotions:
            score = adjusted_emotions[em]
            if self.personal_profiles.get(em):
                calibrated_max = self.personal_profiles[em].get(em, 0.5)
                if natural_baseline:
                    calibrated_max -= natural_baseline.get(em, 0.0)
                calibrated_max = max(0.05, calibrated_max) 
                relative_scores[em] = score / calibrated_max
            else:
                default_thresh = {"Happy": 0.35, "Angry": 0.45, "Sad": 0.30, "Fearful": 0.40}.get(em, 0.35)
                relative_scores[em] = score / default_thresh
                
        max_emotion = max(relative_scores, key=relative_scores.get)
        max_rel_score = relative_scores[max_emotion]
        
        if max_rel_score < 0.50:
            return "Natural (Neutral)", adjusted_emotions
            
        return max_emotion, adjusted_emotions

    def _get_head_pose(self, landmarks, frame_width, frame_height):
        image_points = np.array([
            (landmarks[1].x * frame_width, landmarks[1].y * frame_height),
            (landmarks[152].x * frame_width, landmarks[152].y * frame_height),
            (landmarks[33].x * frame_width, landmarks[33].y * frame_height),
            (landmarks[263].x * frame_width, landmarks[263].y * frame_height),
            (landmarks[61].x * frame_width, landmarks[61].y * frame_height),
            (landmarks[291].x * frame_width, landmarks[291].y * frame_height)
        ], dtype=np.float64)

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(self.face_3d_model, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        return angles[0], angles[1], angles[2]

    def _get_gaze_ratio(self, eye_landmarks, iris_landmark, frame_width, frame_height):
        p_left = np.array([eye_landmarks[0].x * frame_width, eye_landmarks[0].y * frame_height])
        p_right = np.array([eye_landmarks[1].x * frame_width, eye_landmarks[1].y * frame_height])
        p_top = np.array([eye_landmarks[2].x * frame_width, eye_landmarks[2].y * frame_height])
        p_bottom = np.array([eye_landmarks[3].x * frame_width, eye_landmarks[3].y * frame_height])
        p_iris = np.array([iris_landmark.x * frame_width, iris_landmark.y * frame_height])
        
        horiz_dist = max(np.linalg.norm(p_right - p_left), 0.001)
        vert_dist = max(np.linalg.norm(p_bottom - p_top), 0.001)
        
        horiz_ratio = np.linalg.norm(p_iris - p_left) / horiz_dist
        vert_ratio = np.linalg.norm(p_iris - p_top) / vert_dist
        return horiz_ratio, vert_ratio

    # --- Processing Logic ---
    def process_frame_base64(self, b64_string):
        """Processes a single base64 image frame sent via API."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            image_data = base64.b64decode(b64_string)
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")
        except Exception as e:
            return {"error": f"Invalid base64 image: {str(e)}"}

        current_time = time.time()
        with self.lock:
            calib_mode_current = self.calibration_mode
            is_auto_calibrating = not self.is_center_calibrated

        self.last_process_time = current_time

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_landmarker.detect(mp_image)

        h, w, _ = frame.shape
        status = "Focused"
        current_emotion = "Natural (Neutral)"
        display_scores = {}
        distracted = False
        reasons = []

        if detection_result.face_landmarks and detection_result.face_blendshapes:
            landmarks = detection_result.face_landmarks[0]
            blendshapes = detection_result.face_blendshapes[0]

            raw_emotions = self._get_raw_emotion_scores(blendshapes)
            pitch, yaw, roll = self._get_head_pose(landmarks, w, h)

            left_eye_landmarks = [landmarks[33], landmarks[133], landmarks[159], landmarks[145]]
            iris_landmark = landmarks[468]
            horiz_ratio, vert_ratio = self._get_gaze_ratio(left_eye_landmarks, iris_landmark, w, h)

            with self.lock:
                if calib_mode_current:
                    elapsed = time.time() - self.calibration_start_time
                    remaining = int(self.calibration_duration - elapsed)

                    if remaining >= 0:
                        self.calibration_data_buffer.append(raw_emotions)
                        if calib_mode_current == 'Natural':
                            self.calibration_distraction_buffer.append({'pitch': pitch, 'yaw': yaw, 'horiz': horiz_ratio, 'vert': vert_ratio})
                    else:
                        if len(self.calibration_data_buffer) > 0:
                            self.personal_profiles[calib_mode_current] = {
                                "Happy": np.mean([d["Happy"] for d in self.calibration_data_buffer]),
                                "Angry": np.mean([d["Angry"] for d in self.calibration_data_buffer]),
                                "Sad": np.mean([d["Sad"] for d in self.calibration_data_buffer]),
                                "Fearful": np.mean([d["Fearful"] for d in self.calibration_data_buffer])
                            }
                        if calib_mode_current == 'Natural' and len(self.calibration_distraction_buffer) > 0:
                            self.center_baseline = {
                                'pitch': np.mean([d["pitch"] for d in self.calibration_distraction_buffer]),
                                'yaw': np.mean([d["yaw"] for d in self.calibration_distraction_buffer]),
                                'horiz': np.mean([d["horiz"] for d in self.calibration_distraction_buffer]),
                                'vert': np.mean([d["vert"] for d in self.calibration_distraction_buffer])
                            }
                            self.is_center_calibrated = True
                        self.calibration_mode = None

                # Auto Calibrate Center
                elif not self.is_center_calibrated:
                    self.auto_center_buffer.append({'pitch': pitch, 'yaw': yaw, 'horiz': horiz_ratio, 'vert': vert_ratio})
                    if len(self.auto_center_buffer) >= self.auto_calibrate_frames:
                        self.center_baseline = {
                            'pitch': np.mean([d["pitch"] for d in self.auto_center_buffer]),
                            'yaw': np.mean([d["yaw"] for d in self.auto_center_buffer]),
                            'horiz': np.mean([d["horiz"] for d in self.auto_center_buffer]),
                            'vert': np.mean([d["vert"] for d in self.auto_center_buffer])
                        }
                        self.is_center_calibrated = True

                current_emotion, display_scores = self._detect_emotion_calibrated(raw_emotions)

                pitch_diff = pitch - self.center_baseline['pitch']
                yaw_diff = yaw - self.center_baseline['yaw']
                horiz_diff = horiz_ratio - self.center_baseline['horiz']
                vert_diff = vert_ratio - self.center_baseline['vert']

                if pitch_diff > 10: reasons.append("Head Up")
                elif pitch_diff < -10: reasons.append("Head Down")
                if yaw_diff > 10: reasons.append("Head Right")
                elif yaw_diff < -10: reasons.append("Head Left")
                if horiz_diff < -0.08: reasons.append("Eyes Left")
                elif horiz_diff > 0.08: reasons.append("Eyes Right")
                if vert_diff < -0.08: reasons.append("Eyes Up")
                elif vert_diff > 0.08: reasons.append("Eyes Down")

                if reasons:
                    distracted = True
                    status = "Distracted"

                # Aggregation Logic
                if self.session_active and not calib_mode_current and self.is_center_calibrated:
                    self.stats["total_samples"] += 1
                    if distracted:
                        self.stats["distracted_samples"] += 1

                    if current_emotion in self.stats["emotions"]:
                        self.stats["emotions"][current_emotion] += 1

        with self.lock:
            self.latest_state = {
                "emotion": current_emotion,
                "status": status,
                "distraction_reasons": reasons,
                "emotion_scores": {k: float(v) for k, v in display_scores.items()},
                "is_calibrating": self.calibration_mode is not None,
                "calibration_mode": self.calibration_mode,
                "timestamp": current_time,
                "session_start_time": self.session_start_time if self.session_active else None,
                "total_samples": self.stats["total_samples"] if self.session_active else 0,
                "distracted_samples": self.stats["distracted_samples"] if self.session_active else 0
            }
        
        return self.get_latest_state()
