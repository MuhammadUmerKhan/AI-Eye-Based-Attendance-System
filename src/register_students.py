import numpy as np
import warnings
from typing import Tuple
from src.extract_embeddings import extract_embedding
from src.logger import get_logger
from src.faiss_index import FaissIndex
from src.config import EMBEDDING_DIM

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = get_logger(__name__)

def register_student(db, student_id: int, name: str, course_id: str, course_name: str, image_input, faiss_index: FaissIndex) -> Tuple[bool, str]:
    """
    Register a student by saving their ID, name, course details, and eye region embedding to the database,
    and update the FAISS index for the specific course.
    Accepts a Database instance, image input (file path or file-like object), and a FaissIndex instance.
    Returns success status and message.
    """
    try:
        logger.debug(f"Registering student {student_id}, name: {name}, course_id: {course_id}")
        if db.check_duplicate(student_id, course_id):
            logger.error({"student_id": student_id, "course_id": course_id, "message": f"Student {student_id} already registered in course {course_id}"})
            return False, f"Student {student_id} is already registered in course {course_name}"
        
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning({"error": error, "message": f"Registration failed for student {student_id} in course {course_id}"})
            return False, error
        
        if embedding.shape[0] != EMBEDDING_DIM:
            logger.error({"got": embedding.shape[0], "expected": EMBEDDING_DIM, "message": f"Unexpected embedding dimension for student {student_id}"})
            return False, f"Unexpected embedding dimension {embedding.shape[0]}, expected {EMBEDDING_DIM}"
        
        success, message = db.register_student(student_id, name, course_id, course_name, embedding)
        if not success:
            logger.error({"student_id": student_id, "course_id": course_id, "message": message})
            return False, message
        
        embedding = embedding.astype(np.float32)
        faiss_index.update_index(embedding, student_id, name, course_id)
        logger.info({"student_id": student_id, "course_id": course_id, "message": f"{message} and FAISS index updated"})
        return True, f"{message} and FAISS index updated"
    except Exception as e:
        logger.error({"error": str(e), "message": f"Error registering student {student_id} in course {course_id}"})
        return False, f"Error: {str(e)}"