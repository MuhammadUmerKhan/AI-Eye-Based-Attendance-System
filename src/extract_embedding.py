import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from config import DEEPFACE_MODEL

def preprocess_eye_region(img):
    """
    Preprocess eye region for consistent embedding extraction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def crop_both_eyes_region_mediapipe(image_path):
    """
    Extract eye region from an image using MediaPipe Face Mesh.
    Returns cropped eye region and bounding box coordinates.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    
    results = face_mesh.process(rgb_img)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        
        # Eye landmark indices for left and right eyes
        left_eye = [33, 133, 160, 159, 158, 157, 173]
        right_eye = [362, 382, 387, 386, 385, 384, 398]
        
        x_coords = [landmark.x * w for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]
        y_coords = [landmark.y * h for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]
        
        x_min = max(0, int(min(x_coords)) - 10)
        x_max = min(w, int(max(x_coords)) + 10)
        y_min = max(0, int(min(y_coords)) - 5)
        y_max = min(h, int(max(y_coords)) + 5)
        
        face_mesh.close()
        return img[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    face_mesh.close()
    return None, None

def extract_embedding(image_path, model_name=DEEPFACE_MODEL):
    """
    Extract embedding from the eye region of an image.
    Returns embedding and error message (if any).
    """
    eye_region, bbox = crop_both_eyes_region_mediapipe(image_path)
    if eye_region is None:
        return None, None, "Failed to detect eye region"
    
    try:
        eye_region = preprocess_eye_region(eye_region)
        embedding = DeepFace.represent(
            img_path=eye_region,
            model_name=model_name,
            enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding), bbox, None
    except Exception as e:
        return None, None, f"Embedding extraction failed: {str(e)}"