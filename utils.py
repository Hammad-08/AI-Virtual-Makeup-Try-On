import numpy as np
import mediapipe as mp
import cv2
import onnxruntime
import os

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Enhanced face points mapping from allmakeup.py
face_points = {
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    
    # Improved eyeliner points for more precise application
    "EYELINER_LEFT_UPPER": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "EYELINER_LEFT_LOWER": [33, 130, 226, 113, 225, 224, 223, 222, 221, 189, 244, 245, 122, 6, 202, 214, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    "EYELINER_RIGHT_UPPER": [362, 382, 381, 380, 374, 373, 390, 249, 263],
    "EYELINER_RIGHT_LOWER": [362, 359, 446, 342, 445, 444, 443, 442, 441, 418, 464, 465, 351, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 375, 321, 308, 324, 318],
    
    # Better eyeshadow regions
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    
    # Enhanced eyebrow regions
    "EYEBROW_LEFT": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70],
    "EYEBROW_RIGHT": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 300],
    
    # Original face points for compatibility
    "EYELINER_LEFT": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "EYELINER_RIGHT": [362, 382, 381, 380, 374, 373, 390, 249, 263],
}

class EnhancedVirtualMakeup:
    def __init__(self, model_path=None):
        # Try to load ONNX model
        self.session = None
        self.has_onnx_model = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.session = onnxruntime.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]
                self.output_name = self.session.get_outputs()[0].name
                self.has_onnx_model = True
                print(f"ONNX model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Could not load ONNX model: {e}")
                self.session = None
        else:
            print("ONNX model path not provided or file doesn't exist, using fallback methods")
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Default makeup settings
        self.foundation_intensity = 0.3
        self.foundation_color = [255, 224, 189]  # BGR
        self.debug_mode = False

    def preprocess(self, image):
        """Preprocess image for ONNX model"""
        if not self.has_onnx_model:
            return None
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        normalized = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        transposed = normalized.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def get_skin_mask(self, image):
        """Get skin segmentation mask using ONNX model or fallback"""
        if self.has_onnx_model:
            try:
                input_tensor = self.preprocess(image)
                output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
                mask = np.squeeze(output).transpose(1, 2, 0)
                skin_mask = cv2.resize(mask[:, :, 0], (image.shape[1], image.shape[0]))
                hair_mask = cv2.resize(mask[:, :, 2], (image.shape[1], image.shape[0]))
                
                # Convert to binary masks
                skin_binary = (skin_mask > 0.3).astype(np.uint8)
                hair_binary = (hair_mask > 0.5).astype(np.uint8)
                
                if self.debug_mode:
                    print(f"ONNX Skin coverage: {(np.sum(skin_binary)/skin_binary.size)*100:.1f}%")
                
                return skin_binary, hair_binary
            except Exception as e:
                print(f"ONNX inference error: {e}")
                return self.get_fallback_skin_mask(image)
        else:
            return self.get_fallback_skin_mask(image)

    def get_fallback_skin_mask(self, image):
        """Fallback skin mask using face landmarks"""
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Create face region mask
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use a combination of face landmarks to create face outline
        face_outline_indices = list(range(0, min(468, len(landmarks))))[:100:2]  # Sample face points
        
        face_points_coords = []
        for idx in face_outline_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                face_points_coords.append((x, y))
        
        if len(face_points_coords) >= 3:
            # Create convex hull for face region
            face_points_array = np.array(face_points_coords, dtype=np.int32)
            hull = cv2.convexHull(face_points_array)
            cv2.fillPoly(face_mask, [hull], 255)
        
        # Create simple hair mask (upper portion of face)
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        if len(face_points_coords) >= 3:
            # Simple hair region - upper 30% of face bounding box
            min_y = min([p[1] for p in face_points_coords])
            max_y = max([p[1] for p in face_points_coords])
            hair_height = int((max_y - min_y) * 0.3)
            hair_mask[max(0, min_y-hair_height):min_y, :] = face_mask[max(0, min_y-hair_height):min_y, :]
        
        return (face_mask > 0).astype(np.uint8), (hair_mask > 0).astype(np.uint8)

    def get_face_landmarks(self, frame):
        """Extract face landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

    def get_region_mask(self, landmarks, indices, frame_shape):
        """Create region mask from landmark indices"""
        h, w = frame_shape[:2]
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                         for i in indices if i < len(landmarks)], dtype=np.int32)
        if len(points) >= 3:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            return mask
        return np.zeros((h, w), dtype=np.uint8)

    def apply_smooth_color(self, frame, mask, color, intensity, blur_kernel=15):
        """Apply color with smooth blending and edge softening"""
        if np.sum(mask) == 0:
            return frame
            
        # Create smooth mask with Gaussian blur for natural edges
        smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0)
        smooth_mask = smooth_mask / 255.0
        
        # Apply color blending
        result = frame.copy().astype(np.float32)
        color_array = np.array(color, dtype=np.float32)
        
        for c in range(3):
            channel_mask = smooth_mask * intensity
            result[:, :, c] = result[:, :, c] * (1 - channel_mask) + color_array[c] * channel_mask
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_foundation(self, frame, color, intensity=0.3):
        """Apply foundation using ONNX skin segmentation or fallback"""
        landmarks = self.get_face_landmarks(frame)
        if landmarks is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # Get skin mask using ONNX or fallback
        skin_mask, _ = self.get_skin_mask(frame)
        
        if np.sum(skin_mask) == 0:
            if self.debug_mode:
                print("WARNING: No skin detected! Foundation cannot be applied.")
            return frame
        
        # Create exclusion mask for eyes and lips
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Exclude eye areas
        left_eye_mask = self.get_region_mask(landmarks, face_points["LEFT_EYE"], frame.shape)
        right_eye_mask = self.get_region_mask(landmarks, face_points["RIGHT_EYE"], frame.shape)
        
        # Exclude lip areas
        upper_lip_mask = self.get_region_mask(landmarks, face_points["LIP_UPPER"], frame.shape)
        lower_lip_mask = self.get_region_mask(landmarks, face_points["LIP_LOWER"], frame.shape)
        
        # Exclude eye brow
        eyebrow_left = self.get_region_mask(landmarks, face_points["EYEBROW_LEFT"], frame.shape)
        eyebrow_right = self.get_region_mask(landmarks, face_points["EYEBROW_RIGHT"], frame.shape)
        
        exclusion_mask = left_eye_mask | right_eye_mask | upper_lip_mask | lower_lip_mask | eyebrow_left | eyebrow_right
        
        # Foundation mask (skin but not excluded areas)
        foundation_mask = (skin_mask == 1) & (exclusion_mask == 0)
        foundation_mask = foundation_mask.astype(np.uint8) * 255
        
        if self.debug_mode:
            print(f"Applying foundation to {np.sum(foundation_mask > 0)} pixels")
        
        return self.apply_smooth_color(frame, foundation_mask, color, intensity, blur_kernel=21)

    def set_foundation_settings(self, color, intensity):
        """Set foundation color and intensity"""
        self.foundation_color = color
        self.foundation_intensity = intensity

# Create global instance
# Try common ONNX model paths
possible_model_paths = [
    "model/best_model_simplifier.onnx"
]

model_path = None
for path in possible_model_paths:
    if os.path.exists(path):
        model_path = path
        break

enhanced_makeup = EnhancedVirtualMakeup(model_path)

def read_landmarks(image: np.array):
    """Extract face landmarks - compatibility function"""
    landmarks = enhanced_makeup.get_face_landmarks(image)
    if landmarks is None:
        return None
    
    # Convert to dictionary format for compatibility
    h, w = image.shape[:2]
    landmark_coordinates = {}
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Ensure coordinates are within image bounds
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        landmark_coordinates[idx] = (x, y)
    
    return landmark_coordinates

def add_mask(mask: np.array, idx_to_coordinates: dict, face_connections: list, colors: list):
    """Create mask for facial features with colors"""
    for i, connection in enumerate(face_connections):
        if i >= len(colors):
            break
            
        # Filter out landmarks not found in the coordinates
        points = [idx_to_coordinates[idx] for idx in connection if idx in idx_to_coordinates]
        if len(points) < 3:
            continue

        # Create a shape for the feature on the mask and add color
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], colors[i])

    # Smoothen the image
    mask = cv2.GaussianBlur(mask, (7, 7), 4)
    return mask

def apply_foundation_to_frame(frame, color, intensity=0.3):
    """Apply foundation using enhanced processor"""
    return enhanced_makeup.apply_foundation(frame, color, intensity)

def process_frame(frame, face_connections, colors):
    """Process a single frame from the video feed"""
    # Create an empty mask like the frame
    mask = np.zeros_like(frame)

    # Extract facial landmarks
    face_landmarks = read_landmarks(image=frame)

    # If no landmarks are detected, display "No face found" on the frame
    if face_landmarks is None:
        cv2.putText(
            frame,
            "No face found",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    # Create mask for facial features with colors
    if face_connections and colors:
        mask = add_mask(
            mask,
            idx_to_coordinates=face_landmarks,
            face_connections=face_connections,
            colors=colors,
        )

        # Combine the original frame and the mask with weights
        output = cv2.addWeighted(frame, 1.0, mask, 0.4, 1.0)
        return output
    
    return frame
