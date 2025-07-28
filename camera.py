import cv2
import numpy as np
from utils import *
from PySide6.QtCore import QThread, Signal, QMutex
from PySide6.QtGui import QImage
from collections import deque
import time

# Enhanced face elements including foundation
face_elements = [
    "FOUNDATION",
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
]

# Enhanced colors map with foundation and intensity
colors_map = {
    "FOUNDATION": {"color": [], "intensity": 0.3},
    "LIP_UPPER": {"color": [], "intensity": 1},
    "LIP_LOWER": {"color": [], "intensity": 1},
    "EYELINER_LEFT": {"color": [], "intensity": 0.8},
    "EYELINER_RIGHT": {"color": [], "intensity": 0.8},
    "EYESHADOW_LEFT": {"color": [], "intensity": 0.6},
    "EYESHADOW_RIGHT": {"color": [], "intensity": 0.6},
    "EYEBROW_LEFT": {"color": [], "intensity": 0.7},
    "EYEBROW_RIGHT": {"color": [], "intensity": 0.7},
}

class CameraThread(QThread):
    frame_captured = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.cap = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize camera with DroidCam USB (no WiFi/IP attempts)"""
        try:
            # Method 1: Try common DroidCam USB indices directly
            if self.try_common_droidcam_indices():
                return
                
            # Method 2: Try extended range for DroidCam
            if self.try_extended_camera_range():
                return
                
            # Method 3: Try regular cameras as fallback
            self.try_regular_cameras()
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def try_common_droidcam_indices(self):
        """Try common DroidCam USB camera indices"""
        print("Trying DroidCam USB connection...")
        
        # Most common indices where DroidCam USB appears
        # Usually DroidCam appears at index 1, 2, or 3 when USB connected
        common_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        for camera_index in common_indices:
            print(f"Testing camera index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Found working camera at index {camera_index}! Resolution: {width}x{height}")
                    self.setup_camera_properties()
                    return True
                else:
                    print(f"✗ Camera {camera_index} opened but no frame received")
                    self.cap.release()
                    self.cap = None
            else:
                print(f"✗ Camera {camera_index} failed to open")
                if self.cap:
                    self.cap.release()
                self.cap = None
        
        print("No DroidCam found in common indices")
        return False

    def try_extended_camera_range(self):
        """Try extended range for DroidCam detection"""
        print("Trying extended camera range...")
        
        # Extended range - sometimes DroidCam appears at higher indices
        extended_indices = list(range(11, 20))
        
        for camera_index in extended_indices:
            print(f"Testing camera index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Found working camera at index {camera_index}! Resolution: {width}x{height}")
                    self.setup_camera_properties()
                    return True
                else:
                    self.cap.release()
                    self.cap = None
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None
        
        return False

    def setup_camera_properties(self):
        """Setup optimal camera properties for DroidCam"""
        try:
            # Set preferred resolution (DroidCam supports various resolutions)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 100)
            
            # Reduce buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Additional settings that might help with DroidCam
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            print("Camera properties configured for DroidCam")
            
        except Exception as e:
            print(f"Error setting camera properties: {e}")

    def try_regular_cameras(self):
        """Try regular camera indices as fallback"""
        print("Trying regular cameras (including index 0)...")
        
        # Include index 0 as sometimes DroidCam might appear there too
        regular_indices = [0, 1, 2, 3, 4]
        
        for camera_index in regular_indices:
            print(f"Testing regular camera index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✓ Camera {camera_index} working! Resolution: {width}x{height}")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 100)
                    return True
                else:
                    self.cap.release()
                    self.cap = None
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None
        
        print("ERROR: No working camera found!")
        return False

    def run(self):
        print("Camera thread started")
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.initialize_camera()
                    if self.cap is None:
                        # Send a black frame with message
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(black_frame, "No Camera Found", (200, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        self.frame_captured.emit(black_frame)
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cv2.flip(frame, 1)  # Mirror effect
                    self.frame_captured.emit(frame)
                else:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    
            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.1)

    def stop(self):
        print("Stopping camera thread...")
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        self.quit()
        print("Camera thread stopped")

class ProcessingThread(QThread):
    frame_ready = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.input_frame = None
        self.mutex = QMutex()
        self.face_connections = []
        self.colors = []
        self.intensities = []
        self.foundation_settings = {"color": [255, 224, 189], "intensity": 0.3}

        # Store last frames for smoothing
        self.previous_landmarks = deque(maxlen=3)

    def update_frame(self, frame):
        if not self.running:
            return
        try:
            self.mutex.lock()
            self.input_frame = frame.copy() if frame is not None else None
        finally:
            self.mutex.unlock()

    def smooth_landmarks(self, new_landmarks):
        if not new_landmarks:
            return None

        if not self.previous_landmarks:
            self.previous_landmarks.append(new_landmarks)
            return new_landmarks

        # Simple smoothing
        self.previous_landmarks.append(new_landmarks)
        return new_landmarks

    def clear_all_makeup(self):
        global colors_map
        for key in colors_map.keys():
            colors_map[key] = {"color": [], "intensity": 0.7}
        self.face_connections = []
        self.colors = []
        self.intensities = []
        print("Cleared all makeup")

    def remove_makeup(self, face_elements):
        global colors_map
        if face_elements:
            for face_element in face_elements:
                if face_element in colors_map:
                    colors_map[face_element] = {"color": [], "intensity": 0.7}
                    print(f"Removed {face_element}")

        # Update face connections and colors
        self.update_face_connections()

    def update_makeup(self, face_elements, color_code, intensity=0.7):
        global colors_map
        try:
            # Convert color_code to BGR
            bgr_color = tuple(int(color_code.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))
            # print(f"Applying makeup: {face_elements} with color {color_code} -> {bgr_color}, intensity: {intensity}")

            # Update the colors_map for the provided face elements
            for face_element in face_elements:
                colors_map[face_element] = {"color": bgr_color, "intensity": intensity}
                
                # Special handling for foundation
                if face_element == "FOUNDATION":
                    self.foundation_settings = {"color": list(bgr_color), "intensity": intensity}
                    enhanced_makeup.set_foundation_settings(list(bgr_color), intensity)
                    # print(f"Foundation settings updated: {self.foundation_settings}")

            # Update face connections and colors
            self.update_face_connections()
            
        except Exception as e:
            print(f"Error updating makeup: {e}")

    def update_product_intensity(self, face_elements, color_code, intensity):
        """Update intensity for specific product"""
        global colors_map
        try:
            bgr_color = tuple(int(color_code.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))
            # print(f"Updating intensity for {face_elements}: {intensity}")

            for face_element in face_elements:
                if face_element in colors_map and colors_map[face_element]["color"]:
                    colors_map[face_element]["intensity"] = intensity
                    
                    # Special handling for foundation
                    if face_element == "FOUNDATION":
                        self.foundation_settings["intensity"] = intensity
                        enhanced_makeup.set_foundation_settings(
                            self.foundation_settings["color"], 
                            intensity
                        )

            # Update face connections and colors
            self.update_face_connections()
            
        except Exception as e:
            print(f"Error updating intensity: {e}")

    def update_face_connections(self):
        """Update face connections and colors based on current colors_map"""
        global colors_map
        self.face_connections = []
        self.colors = []
        self.intensities = []
        
        for element in colors_map.keys():
            if element != "FOUNDATION" and colors_map[element]["color"] and element in face_points:
                self.face_connections.append(face_points[element])
                self.colors.append(colors_map[element]["color"])
                self.intensities.append(colors_map[element]["intensity"])
        
        print(f"Active makeup elements: {len(self.face_connections)}")

    def run(self):
        print("Processing thread started")
        
        while self.running:
            try:
                if self.input_frame is not None:
                    self.mutex.lock()
                    frame = self.input_frame.copy()
                    self.mutex.unlock()
                    
                    processed_frame = self._process_frame(frame)
                    
                    if processed_frame is not None and self.running:
                        height, width, channel = processed_frame.shape
                        bytes_per_line = channel * width
                        q_image = QImage(
                            processed_frame.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_BGR888,
                        )
                        self.frame_ready.emit(q_image)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Processing thread error: {e}")
                time.sleep(0.1)

    def _process_frame(self, frame):
        try:
            if frame is None:
                return None
                
            result = frame.copy()
            
            # Apply foundation first if enabled using ONNX model
            if colors_map.get("FOUNDATION", {}).get("color"):
                result = apply_foundation_to_frame(
                    result, 
                    self.foundation_settings["color"], 
                    self.foundation_settings["intensity"]
                )
            
            # Apply other makeup effects with individual intensities
            if self.face_connections and self.colors and self.intensities:
                result = self.process_frame_with_intensity(result, self.face_connections, self.colors, self.intensities)
            else:
                # Still check for face detection even without makeup
                face_landmarks = read_landmarks(image=result)
                if face_landmarks is None:
                    pass
                    # cv2.putText(
                    #     result,
                    #     "No face found",
                    #     (50, 50),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (0, 0, 255),
                    #     2,
                    #     cv2.LINE_AA,
                    # )
            
            return result
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

    def process_frame_with_intensity(self, frame, face_connections, colors, intensities):
        """Process frame with individual intensity controls"""
        # Create an empty mask like the frame
        mask = np.zeros_like(frame)

        # Extract facial landmarks
        face_landmarks = read_landmarks(image=frame)

        # If no landmarks are detected, display "No face found" on the frame
        if face_landmarks is None:
            # cv2.putText(
            #     frame,
            #     "No face found",
            #     (50, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 0, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            return frame

        # Create mask for facial features with colors and individual intensities
        if face_connections and colors and intensities:
            for i, connection in enumerate(face_connections):
                if i >= len(colors) or i >= len(intensities):
                    break
                    
                # Filter out landmarks not found in the coordinates
                points = [face_landmarks[idx] for idx in connection if idx in face_landmarks]
                if len(points) < 3:
                    continue

                # Create individual mask for this feature
                feature_mask = np.zeros_like(frame)
                points_array = np.array(points, dtype=np.int32)
                cv2.fillPoly(feature_mask, [points_array], colors[i])
                
                # Apply Gaussian blur
                feature_mask = cv2.GaussianBlur(feature_mask, (7, 7), 4)
                
                # Blend with custom intensity
                intensity = intensities[i]
                mask = cv2.addWeighted(mask, 1.0, feature_mask, intensity, 0)

            # Combine the original frame and the mask
            if np.any(mask):
                output = cv2.addWeighted(frame, 1.0, mask, 0.4, 1.0)
                return output
        
        return frame

    def stop(self):
        print("Stopping processing thread...")
        self.running = False
        self.quit()
        print("Processing thread stopped")