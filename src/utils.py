import os
import cv2
import numpy as np
from src.logger import get_logger

# Configure logging
logger = get_logger(__name__)

def save_image(file, output_path):
    """
    Save an uploaded image to the specified path.
    Returns success status and message.
    """
    try:
        logger.debug(f"Saving image to {output_path}")
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            logger.error({"message": f"Failed to decode image from {output_path}"})
            return False, "Failed to decode image"
        
        logger.debug(f"Decoded image shape: {img.shape}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        logger.info({"message": f"Image saved to {output_path}"})
        return True, f"Image saved to {output_path}"
    except Exception as e:
        logger.error({"error": str(e), "message": f"Failed to save image to {output_path}"})
        return False, f"Failed to save image: {str(e)}"

def cleanup_temp_image(image_path):
    """
    Delete a temporary image file.
    """
    try:
        logger.debug(f"Attempting to delete temporary image: {image_path}")
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info({"message": f"Temporary image deleted: {image_path}"})
        else:
            logger.warning({"message": f"Temporary image not found: {image_path}"})
    except Exception as e:
        logger.error({"error": str(e), "message": f"Failed to delete temporary image {image_path}"})