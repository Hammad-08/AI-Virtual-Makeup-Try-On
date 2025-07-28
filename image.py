import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import math

# Load ONNX model
MODEL_PATH = "C:/workspace/virtual-makeup/models/pretrained/best_model_simplifier.onnx"

class VirtualMakeup:
    def __init__(self):
        self.session = onnxruntime.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.output_name = self.session.get_outputs()[0].name
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # MAKEUP SETTINGS
        self.foundation_intensity = 0.8
        self.foundation_color = [ 72, 127, 160]  # BGR
        
        # Lipstick settings
        self.lipstick_intensity = 0.7
        self.lipstick_color = [40, 60, 180]  # BGR - Red lipstick
        
        # Eyeliner settings
        self.eyeliner_intensity = 0.8
        self.eyeliner_color = [20, 20, 20]  # BGR - Black eyeliner
        self.eyeliner_thickness = 2
        
        # Eyeshadow settings
        self.eyeshadow_intensity = 0.6
        self.eyeshadow_color = [120, 80, 150]  # BGR - Purple eyeshadow
        
        # Eyebrow settings
        self.eyebrow_intensity = 0.7
        self.eyebrow_color = [40, 30, 25]  # BGR - Dark brown eyebrows
        
        self.debug_mode = True
        
        # Enhanced face points mapping with better eyebrow coverage
        self.face_points = {
            "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],
            "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
            
            # Improved eyeliner points for more precise application
            "EYELINER_LEFT_UPPER": [33, 7, 163, 144, 145, 153, 154, 155, 133],
            "EYELINER_LEFT_LOWER": [33, 130, 226, 113, 225, 224, 223, 222, 221, 189, 244, 245, 122, 6, 202, 214, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            "EYELINER_RIGHT_UPPER": [362, 382, 381, 380, 374, 373, 390, 249, 263],
            "EYELINER_RIGHT_LOWER": [362, 359, 446, 342, 445, 444, 443, 442, 441, 418, 464, 465, 351, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 375, 321, 308, 324, 318],
            
            # Better eyeshadow regions
            "EYESHADOW_LEFT": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "EYESHADOW_RIGHT": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            
            "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
            "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
            
            # Enhanced eyebrow regions with hair strokes
            "EYEBROW_LEFT_FULL": [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 308, 324, 318, 297, 332, 284, 298, 293, 334, 296, 336, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107, 55, 65, 46],
            "EYEBROW_RIGHT_FULL": [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276],
            "EYEBROW_LEFT": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70],
            "EYEBROW_RIGHT": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 300]
        }

    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        normalized = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        transposed = normalized.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def infer(self, image):
        input_tensor = self.preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        mask = np.squeeze(output).transpose(1, 2, 0)
        skin_mask = cv2.resize(mask[:, :, 0], (image.shape[1], image.shape[0]))
        hair_mask = cv2.resize(mask[:, :, 2], (image.shape[1], image.shape[0]))
        
        # Convert to binary masks
        skin_binary = (skin_mask > 0.3).astype(np.uint8)
        hair_binary = (hair_mask > 0.5).astype(np.uint8)
        
        if self.debug_mode:
            print(f"Skin mask stats: min={skin_mask.min():.3f}, max={skin_mask.max():.3f}, mean={skin_mask.mean():.3f}")
            print(f"Skin pixels detected: {np.sum(skin_binary)} out of {skin_binary.size} total pixels")
            print(f"Skin coverage: {(np.sum(skin_binary)/skin_binary.size)*100:.1f}%")
        
        return skin_binary, hair_binary

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

    def apply_foundation(self, frame, skin_mask, landmarks):
        """Apply foundation to skin areas"""
        if landmarks is None or np.sum(skin_mask) == 0:
            if self.debug_mode:
                print("WARNING: No skin detected or landmarks missing! Foundation cannot be applied.")
            return frame
            
        h, w = frame.shape[:2]
        
        # Create minimal exclusion mask for eyes and lips
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Exclude eye areas
        left_eye_mask = self.get_region_mask(landmarks, self.face_points["LEFT_EYE"], frame.shape)
        right_eye_mask = self.get_region_mask(landmarks, self.face_points["RIGHT_EYE"], frame.shape)
        
        # Exclude lip areas
        upper_lip_mask = self.get_region_mask(landmarks, self.face_points["LIP_UPPER"], frame.shape)
        lower_lip_mask = self.get_region_mask(landmarks, self.face_points["LIP_LOWER"], frame.shape)
        
        exclusion_mask = left_eye_mask | right_eye_mask | upper_lip_mask | lower_lip_mask
        
        # Foundation mask (skin but not excluded areas)
        foundation_mask = (skin_mask == 1) & (exclusion_mask == 0)
        foundation_mask = foundation_mask.astype(np.uint8) * 255
        
        if self.debug_mode:
            print(f"Applying foundation to {np.sum(foundation_mask > 0)} pixels")
        
        return self.apply_smooth_color(frame, foundation_mask, self.foundation_color, self.foundation_intensity, blur_kernel=21)

    def apply_lipstick(self, frame, landmarks):
        """Apply lipstick to lips"""
        if landmarks is None:
            if self.debug_mode:
                print("WARNING: No landmarks detected! Lipstick cannot be applied.")
            return frame
            
        # Create lip mask
        upper_lip_mask = self.get_region_mask(landmarks, self.face_points["LIP_UPPER"], frame.shape)
        lower_lip_mask = self.get_region_mask(landmarks, self.face_points["LIP_LOWER"], frame.shape)
        lip_mask = upper_lip_mask | lower_lip_mask
        
        if self.debug_mode:
            print(f"Applying lipstick to {np.sum(lip_mask > 0)} pixels")
        
        return self.apply_smooth_color(frame, lip_mask, self.lipstick_color, self.lipstick_intensity, blur_kernel=9)

    def apply_eyeliner(self, frame, landmarks):
        """Apply natural, professional eyeliner"""
        if landmarks is None:
            return frame
            
        h, w = frame.shape[:2]
        result = frame.copy().astype(np.float32)
        
        # Get precise eyelash line points
        left_upper_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                            for i in self.face_points["EYELINER_LEFT_UPPER"] if i < len(landmarks)]
        right_upper_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                             for i in self.face_points["EYELINER_RIGHT_UPPER"] if i < len(landmarks)]
        
        # Apply subtle, close-to-lashline eyeliner
        for eye_points in [left_upper_points, right_upper_points]:
            if len(eye_points) >= 3:
                # Create very thin, natural-looking liner
                for i in range(len(eye_points) - 1):
                    start_point = eye_points[i]
                    end_point = eye_points[i + 1]
                    
                    # Calculate distance for thickness variation
                    distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
                    
                    # Vary thickness: thinner at inner corner, slightly thicker at outer corner
                    if i < len(eye_points) // 2:
                        thickness = max(1, self.eyeliner_thickness - 1)  # Thinner at inner corner
                    else:
                        thickness = self.eyeliner_thickness  # Normal thickness at outer corner
                    
                    # Draw line with anti-aliasing
                    cv2.line(result, start_point, end_point, self.eyeliner_color, 
                            thickness, cv2.LINE_AA)
                
                # Add very subtle wing only at outer corner (much smaller)
                if len(eye_points) > 2:
                    outer_point = eye_points[0] if eye_points == left_upper_points else eye_points[-1]
                    
                    # Smaller, more natural wing
                    wing_length = 3
                    wing_height = 2
                    
                    if eye_points == left_upper_points:
                        wing_point = (outer_point[0] + wing_length, outer_point[1] - wing_height)
                    else:
                        wing_point = (outer_point[0] - wing_length, outer_point[1] - wing_height)
                    
                    cv2.line(result, outer_point, wing_point, self.eyeliner_color, 
                            max(1, self.eyeliner_thickness - 1), cv2.LINE_AA)
        
        # Very subtle blending for natural look
        eyeliner_mask = np.any(result != frame.astype(np.float32), axis=2).astype(np.float32)
        eyeliner_mask = cv2.GaussianBlur(eyeliner_mask, (3, 3), 0)
        
        # Gentle blending with reduced intensity for natural look
        natural_intensity = self.eyeliner_intensity * 0.7  # Reduce intensity for natural look
        
        for c in range(3):
            result[:, :, c] = (frame[:, :, c].astype(np.float32) * (1 - eyeliner_mask * natural_intensity) + 
                              result[:, :, c] * eyeliner_mask * natural_intensity)
        
        if self.debug_mode:
            print(f"Applied natural eyeliner with reduced intensity")
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_eyeshadow(self, frame, landmarks):
        """Apply natural, professional eyeshadow with soft blending"""
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy().astype(np.float32)
        
        # Create more precise eyeshadow regions
        left_eyeshadow_mask = self.get_region_mask(landmarks, self.face_points["EYESHADOW_LEFT"], frame.shape)
        right_eyeshadow_mask = self.get_region_mask(landmarks, self.face_points["EYESHADOW_RIGHT"], frame.shape)
        
        # Apply natural eyeshadow with professional techniques
        for eye_mask in [left_eyeshadow_mask, right_eyeshadow_mask]:
            if np.sum(eye_mask) > 0:
                # Create multiple layers for professional depth
                # Layer 1: Base neutral tone (very subtle)
                base_color = [self.eyeshadow_color[0] + 30, self.eyeshadow_color[1] + 25, self.eyeshadow_color[2] + 20]
                base_mask = cv2.GaussianBlur(eye_mask.astype(np.float32), (21, 21), 0) / 255.0
                
                # Apply base with very low intensity
                base_intensity = self.eyeshadow_intensity * 0.3
                for c in range(3):
                    result[:, :, c] = (result[:, :, c] * (1 - base_mask * base_intensity) + 
                                     base_color[c] * base_mask * base_intensity)
                
                # Layer 2: Transition shade (medium intensity, smaller area)
                transition_mask = cv2.erode(eye_mask, np.ones((5,5), np.uint8), iterations=1)
                transition_mask = cv2.GaussianBlur(transition_mask.astype(np.float32), (15, 15), 0) / 255.0
                
                transition_intensity = self.eyeshadow_intensity * 0.5
                for c in range(3):
                    result[:, :, c] = (result[:, :, c] * (1 - transition_mask * transition_intensity) + 
                                     self.eyeshadow_color[c] * transition_mask * transition_intensity)
                
                # Layer 3: Deepest shade (lowest intensity, smallest area - mobile lid only)
                deep_mask = cv2.erode(eye_mask, np.ones((8,8), np.uint8), iterations=2)
                deep_mask = cv2.GaussianBlur(deep_mask.astype(np.float32), (9, 9), 0) / 255.0
                
                # Deeper shade for definition
                deep_color = [max(0, self.eyeshadow_color[0] - 20), 
                             max(0, self.eyeshadow_color[1] - 15), 
                             max(0, self.eyeshadow_color[2] - 10)]
                
                deep_intensity = self.eyeshadow_intensity * 0.4
                for c in range(3):
                    result[:, :, c] = (result[:, :, c] * (1 - deep_mask * deep_intensity) + 
                                     deep_color[c] * deep_mask * deep_intensity)
        
        # Add subtle highlight to inner corners and brow bone
        self._add_eyeshadow_highlights(result, landmarks)
        
        if self.debug_mode:
            total_pixels = np.sum(left_eyeshadow_mask > 0) + np.sum(right_eyeshadow_mask > 0)
            print(f"Applied professional layered eyeshadow to {total_pixels} pixels")
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_eyeshadow_highlights(self, frame, landmarks):
        """Add subtle highlights for professional eyeshadow look"""
        h, w = frame.shape[:2]
        
        # Inner corner highlights
        inner_corners = [33, 362]  # Left and right inner corners
        highlight_color = [220, 200, 180]  # Subtle champagne highlight
        
        for corner_idx in inner_corners:
            if corner_idx < len(landmarks):
                x = int(landmarks[corner_idx].x * w)
                y = int(landmarks[corner_idx].y * h)
                
                # Create small highlight area
                highlight_mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(highlight_mask, (x, y), 8, 1.0, -1)
                highlight_mask = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
                
                # Apply highlight with low intensity
                highlight_intensity = 0.3
                for c in range(3):
                    frame[:, :, c] = (frame[:, :, c] * (1 - highlight_mask * highlight_intensity) + 
                                    highlight_color[c] * highlight_mask * highlight_intensity)

    def apply_eyebrow_color(self, frame, landmarks, hair_mask):
        """Apply color to eyebrow hairs with realistic strokes"""
        if landmarks is None:
            if self.debug_mode:
                print("WARNING: No landmarks detected! Eyebrow coloring cannot be applied.")
            return frame
        
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get eyebrow regions
        left_eyebrow_mask = self.get_region_mask(landmarks, self.face_points["EYEBROW_LEFT"], frame.shape)
        right_eyebrow_mask = self.get_region_mask(landmarks, self.face_points["EYEBROW_RIGHT"], frame.shape)
        
        # Combine with hair mask for more accurate eyebrow detection
        eyebrow_hair_left = left_eyebrow_mask & (hair_mask * 255)
        eyebrow_hair_right = right_eyebrow_mask & (hair_mask * 255)
        
        # If no hair detected in eyebrow area, use full eyebrow region
        if np.sum(eyebrow_hair_left) < 100:
            eyebrow_hair_left = left_eyebrow_mask
        if np.sum(eyebrow_hair_right) < 100:
            eyebrow_hair_right = right_eyebrow_mask
        
        # Apply hair-like strokes to eyebrows
        result = self._apply_eyebrow_strokes(result, landmarks, eyebrow_hair_left, "LEFT")
        result = self._apply_eyebrow_strokes(result, landmarks, eyebrow_hair_right, "RIGHT")
        
        # Also apply base color to the entire eyebrow area
        combined_eyebrow_mask = eyebrow_hair_left | eyebrow_hair_right
        result = self.apply_smooth_color(result, combined_eyebrow_mask, 
                                       self.eyebrow_color, self.eyebrow_intensity * 0.6, blur_kernel=5)
        
        if self.debug_mode:
            total_pixels = np.sum(combined_eyebrow_mask > 0)
            print(f"Applied eyebrow coloring to {total_pixels} pixels")
        
        return result

    def _apply_eyebrow_strokes(self, frame, landmarks, eyebrow_mask, side):
        """Apply realistic hair strokes to eyebrows"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        if np.sum(eyebrow_mask) == 0:
            return result
        
        # Get eyebrow points for stroke direction
        if side == "LEFT":
            brow_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                          for i in self.face_points["EYEBROW_LEFT"] if i < len(landmarks)]
        else:
            brow_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                          for i in self.face_points["EYEBROW_RIGHT"] if i < len(landmarks)]
        
        if len(brow_points) < 3:
            return result
        
        # Find eyebrow region bounds
        mask_points = np.where(eyebrow_mask > 0)
        if len(mask_points[0]) == 0:
            return result
        
        min_y, max_y = np.min(mask_points[0]), np.max(mask_points[0])
        min_x, max_x = np.min(mask_points[1]), np.max(mask_points[1])
        
        # Create hair-like strokes
        stroke_overlay = np.zeros_like(frame)
        
        # Generate multiple thin strokes following eyebrow shape
        num_strokes = max(20, (max_x - min_x) // 3)
        
        for i in range(num_strokes):
            # Random position within eyebrow area
            attempts = 0
            while attempts < 10:
                x = np.random.randint(min_x, max_x + 1)
                y = np.random.randint(min_y, max_y + 1)
                
                if y < h and x < w and eyebrow_mask[y, x] > 0:
                    # Calculate stroke direction based on eyebrow shape
                    if side == "LEFT":
                        angle = np.random.uniform(-0.3, 0.5)  # Slightly upward angle
                    else:
                        angle = np.random.uniform(-0.5, 0.3)  # Slightly upward angle
                    
                    # Create stroke
                    stroke_length = np.random.randint(8, 15)
                    end_x = int(x + stroke_length * np.cos(angle))
                    end_y = int(y + stroke_length * np.sin(angle))
                    
                    # Ensure stroke stays within bounds
                    end_x = max(0, min(w-1, end_x))
                    end_y = max(0, min(h-1, end_y))
                    
                    # Draw thin stroke
                    cv2.line(stroke_overlay, (x, y), (end_x, end_y), 
                            self.eyebrow_color, 1, cv2.LINE_AA)
                    break
                
                attempts += 1
        
        # Blend strokes with original image
        stroke_mask = cv2.cvtColor(stroke_overlay, cv2.COLOR_BGR2GRAY)
        stroke_mask = (stroke_mask > 0).astype(np.float32)
        
        # Apply slight blur for more natural look
        stroke_mask = cv2.GaussianBlur(stroke_mask, (3, 3), 0)
        
        # Blend the strokes
        for c in range(3):
            result[:, :, c] = (result[:, :, c] * (1 - stroke_mask * self.eyebrow_intensity) + 
                              stroke_overlay[:, :, c] * stroke_mask * self.eyebrow_intensity)
        
        return result.astype(np.uint8)

    def apply_makeup(self, frame, skin_mask, hair_mask):
        """Apply all makeup effects in the correct order"""
        result = frame.copy()
        
        # Get face landmarks
        landmarks = self.get_face_landmarks(frame)
        
        if landmarks is None:
            print("WARNING: No face landmarks detected! Makeup cannot be applied.")
            return result
        
        print("Applying makeup in layers...")
        
        # 1. Foundation (base layer)
        result = self.apply_foundation(result, skin_mask, landmarks)
        
        # 2. Eyebrow coloring (before eyeshadow)
        result = self.apply_eyebrow_color(result, landmarks, hair_mask)
        
        # 3. Eyeshadow (before eyeliner)
        result = self.apply_eyeshadow(result, landmarks)
        
        # 4. Eyeliner (on top of eyeshadow)
        result = self.apply_eyeliner(result, landmarks)
        
        # 5. Lipstick (final layer)
        result = self.apply_lipstick(result, landmarks)
        
        return result

    # Configuration methods
    def set_foundation_settings(self, intensity=0.8, color=[100, 150, 200]):
        """Set foundation intensity and color"""
        self.foundation_intensity = max(0.0, min(1.0, intensity))
        self.foundation_color = color
        print(f"Foundation: intensity={self.foundation_intensity}, color={self.foundation_color}")
        
    def set_lipstick_settings(self, intensity=0.7, color=[40, 60, 180]):
        """Set lipstick intensity and color"""
        self.lipstick_intensity = max(0.0, min(1.0, intensity))
        self.lipstick_color = color
        print(f"Lipstick: intensity={self.lipstick_intensity}, color={self.lipstick_color}")
        
    def set_eyeliner_settings(self, intensity=0.8, color=[20, 20, 20], thickness=2):
        """Set eyeliner intensity, color, and thickness"""
        self.eyeliner_intensity = max(0.0, min(1.0, intensity))
        self.eyeliner_color = color
        self.eyeliner_thickness = max(1, thickness)
        print(f"Eyeliner: intensity={self.eyeliner_intensity}, color={self.eyeliner_color}, thickness={self.eyeliner_thickness}")
        
    def set_eyeshadow_settings(self, intensity=0.6, color=[120, 80, 150]):
        """Set eyeshadow intensity and color"""
        self.eyeshadow_intensity = max(0.0, min(1.0, intensity))
        self.eyeshadow_color = color
        print(f"Eyeshadow: intensity={self.eyeshadow_intensity}, color={self.eyeshadow_color}")
    
    def set_eyebrow_settings(self, intensity=0.7, color=[40, 30, 25]):
        """Set eyebrow intensity and color"""
        self.eyebrow_intensity = max(0.0, min(1.0, intensity))
        self.eyebrow_color = color
        print(f"Eyebrows: intensity={self.eyebrow_intensity}, color={self.eyebrow_color}")
        
    def toggle_debug(self, enable=True):
        """Enable/disable debug mode"""
        self.debug_mode = enable

def main():
    image_path = "C:\\New folder\\virtual-makeup\\sample\\sample.png"
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    print("Starting Enhanced Virtual Makeup Application...")
    print(f"Image size: {frame.shape}")
    
    makeup_model = VirtualMakeup()
    
    makeup_model.set_foundation_settings(intensity=0.3, color=[72, 127, 194])  # Light skin-tone foundation
    makeup_model.set_lipstick_settings(intensity=0.4, color=[156, 28, 248])        # Bright red lipstick
    makeup_model.set_eyeliner_settings(intensity=0.8, color=[0, 0, 0], thickness=5)  # Black eyeliner
    makeup_model.set_eyeshadow_settings(intensity=0.8, color=[60, 76, 92])      # Blue eyeshadow
    makeup_model.set_eyebrow_settings(intensity=0, color=[164, 164, 168])        # Brown eyebrows

    # Softer eyebrows
    
    print("Running inference...")
    skin_mask, hair_mask = makeup_model.infer(frame)
    
    print("Applying all makeup effects...")
    output = makeup_model.apply_makeup(frame, skin_mask, hair_mask)
    
    cv2.imshow("orignal", frame)
    
    
    # Show results
    if makeup_model.debug_mode:
        cv2.imshow("Full Makeup Result", output)
        
        # Show difference to highlight makeup application
        diff = cv2.absdiff(frame, output)
        diff_enhanced = cv2.addWeighted(diff, 3.0, diff, 0, 0)
        cv2.imshow("Makeup Changes (Enhanced)", diff_enhanced)
        
        print("Press any key to close windows...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()