import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from logger import get_logger

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
        logger.debug(f"Eye region preprocessing completed successfully, output shape: {result.shape}")
        return result
    except Exception as e:
        logger.error({"error": str(e), "message": "Failed to preprocess eye region"})
        return None

def crop_both_eyes_region_mediapipe(image_path):
    """
    Extract eye region from an image using MediaPipe Face Mesh.
    Returns cropped eye region and bounding box coordinates.
    """
    try:
        logger.debug(f"Loading image from {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error({"message": f"Failed to load image from {image_path}"})
            return None, None
        
        logger.debug(f"Image loaded, shape: {img.shape}")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        logger.debug("Initializing MediaPipe Face Mesh")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

        logger.debug("Processing image with Face Mesh")
        results = face_mesh.process(rgb_img)
        if results.multi_face_landmarks:
            logger.debug("Face landmarks detected")
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = img.shape

            # Eye landmark indices for left and right eyes
            left_eye = [33, 133, 160, 159, 158, 157, 173]
            right_eye = [362, 382, 387, 386, 385, 384, 398]

            logger.debug("Calculating eye region coordinates")
            x_coords = [landmark.x * w for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]
            y_coords = [landmark.y * h for landmark in landmarks for idx in left_eye + right_eye if landmark == landmarks[idx]]

            x_min = max(0, int(min(x_coords)) - 10)
            x_max = min(w, int(max(x_coords)) + 10)
            y_min = max(0, int(min(y_coords)) - 5)
            y_max = min(h, int(max(y_coords)) + 5)

            logger.debug(f"Eye region cropped: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            face_mesh.close()
            cropped_img = img[y_min:y_max, x_min:x_max]
            logger.debug(f"Cropped eye region shape: {cropped_img.shape}")
            return cropped_img, (x_min, y_min, x_max, y_max)
        
        logger.warning({"message": "No face landmarks detected"})
        face_mesh.close()
        return None, None
    except Exception as e:
        logger.error({"error": str(e), "message": f"Failed to crop eye region from {image_path}"})
        return None, None

def extract_embedding(image_path, model_name="ArcFace"):
    """
    Extract embedding from the eye region of an image.
    Returns embedding and error message (if any).
    """
    try:
        logger.info(f"Extracting embedding for image: {image_path}")
        eye_region, bbox = crop_both_eyes_region_mediapipe(image_path)
        if eye_region is None:
            logger.warning({"message": "Failed to detect eye region"})
            return None, None, "Failed to detect eye region"

        logger.debug("Preprocessing eye region for embedding")
        eye_region = preprocess_eye_region(eye_region)
        if eye_region is None:
            logger.warning({"message": "Failed to preprocess eye region"})
            return None, None, "Failed to preprocess eye region"

        logger.debug(f"Extracting embedding with {model_name} model, input shape: {eye_region.shape}")
        embedding = DeepFace.represent(
            img_path=eye_region,
            model_name=model_name,
            enforce_detection=False
        )[0]["embedding"]
        embedding = np.array(embedding)
        logger.info({"message": "Embedding extracted successfully", "embedding_shape": embedding.shape})
        return embedding, bbox, None
    except Exception as e:
        logger.error({"error": str(e), "message": f"Embedding extraction failed for {image_path}"})
        return None, None, f"Embedding extraction failed: {str(e)}"