import numpy as np
import warnings
from typing import Tuple
from extract_embeddings import extract_embedding
from logger import get_logger

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def register_students(db, student_id: str, name: str, department: str, image_input) -> Tuple[bool, str]:
    """
    Register a student by saving their ID, name, department, and eye region embedding to the database.
    Accepts a Database instance and either a file path or a file-like object for image_input.
    Returns success status and message.
    """
    try:
        logger.debug(f"Registering student {student_id}, name: {name}, department: {department}, image_input: {image_input}")
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning({"error": error, "message": f"Registration failed for student {student_id}"})
            return False, error
        
        logger.debug(f"Extracted embedding for student {student_id}, shape: {embedding.shape}")
        success, message = db.register_student(student_id, name, department, embedding)
        logger.info({"student_id": student_id, "message": message, "embedding_shape": embedding.shape if embedding is not None else None})
        return success, message
    except Exception as e:
        logger.error({"error": str(e), "message": f"Error registering student {student_id}"})
        return False, f"Error: {str(e)}"