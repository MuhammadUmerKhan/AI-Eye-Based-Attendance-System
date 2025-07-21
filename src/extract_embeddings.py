import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from src.logger import get_logger
from src.config import DEEPFACE_MODEL

# Configure logging
logger = get_logger(__name__)

def preprocess_eye_region(img):
    """
    Preprocess eye region for consistent embedding extraction.
    """
    try:
        logger.debug("Starting eye region preprocessing")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        logger.debug(f"Eye region preprocessing completed, shape: {result.shape}")
        return result
    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to preprocess eye region"})
        return None

def crop_both_eyes_region_mediapipe(image_input):
    """
    Extract eye region from an image using MediaPipe Face Mesh.
    Returns cropped eye region and bounding box coordinates.
    """
    try:
        if isinstance(image_input, str):
            logger.debug(f"Loading image from {image_input}")
            img = cv2.imread(image_input)
        else:
            logger.debug("Loading image from file-like object")
            file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error({"message": "Failed to load image"})
            return None, None
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_img)
            if results.multi_face_landmarks:
                logger.debug("Face landmarks detected")
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = img.shape
                left_eye = [33, 133, 160, 159, 158, 157, 173]
                right_eye = [362, 382, 387, 386, 385, 384, 398]
                
                x_coords = [landmark.x * w for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]
                y_coords = [landmark.y * h for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]
                
                x_min = max(0, int(min(x_coords)) - 10)
                x_max = min(w, int(max(x_coords)) + 10)
                y_min = max(0, int(min(y_coords)) - 5)
                y_max = min(h, int(max(y_coords)) + 5)
                
                logger.debug(f"Eye region cropped: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                return img[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
            
            logger.warning({"message": "No face landmarks detected"})
            return None, None
    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to crop eye region"})
        return None, None

def extract_embedding(image_input, model_name=DEEPFACE_MODEL):
    """
    Extract embedding from the eye region of an image.
    Returns embedding and error message (if any).
    """
    try:
        logger.info(f"Extracting embedding for image")
        eye_region, bbox = crop_both_eyes_region_mediapipe(image_input)
        if eye_region is None:
            logger.warning({"message": "Failed to detect eye region"})
            return None, None, "Failed to detect eye region"

        eye_region = preprocess_eye_region(eye_region)
        if eye_region is None:
            logger.warning({"message": "Failed to preprocess eye region"})
            return None, None, "Failed to preprocess eye region"

        embedding = DeepFace.represent(
            img_path=eye_region,
            model_name=model_name,
            enforce_detection=False
        )[0]["embedding"]
        embedding = np.array(embedding)
        logger.info({"message": "Embedding extracted successfully", "embedding_shape": embedding.shape})
        return embedding, bbox, None
    except Exception as e:
        logger.error({"error": str(e), "message": "Embedding extraction failed"})
        return None, None, f"Embedding extraction failed: {str(e)}"