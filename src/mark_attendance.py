import numpy as np
import faiss
from typing import Tuple, Optional
from datetime import datetime
from extract_embeddings import extract_embedding
from config import FAISS_THRESHOLD
from logger import get_logger

logger = get_logger(__name__)

def mark_attendance(db, image_input) -> Tuple[Optional[str], Optional[str], str]:
    """
    Match an image's embedding against stored embeddings and mark attendance.
    Accepts a Database instance and either a file path or a file-like object for image_input.
    Returns student ID, name, and message.
    """
    try:
        embedding, _, error = extract_embedding(image_input)
        if embedding is None:
            logger.warning(f"Attendance marking failed: {error}")
            return None, None, error
        
        students = db.fetch_students()
        if not students:
            logger.warning("No students registered in database")
            return None, None, "No students registered"
        
        student_ids = []
        names = []
        embeddings = []
        for student in students:
            student_ids.append(student['id'])
            names.append(student['name'])
            embeddings.append(np.frombuffer(student['embedding'], dtype=np.float32))
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Search for nearest neighbor
        D, I = index.search(np.array([embedding], dtype=np.float32), k=1)
        distance = D[0][0]
        
        if distance < FAISS_THRESHOLD:
            student_id = student_ids[I[0][0]]
            name = names[I[0][0]]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success, message = db.mark_attendance(student_id, timestamp)
            if success:
                logger.info(f"Attendance marked for {name} (ID: {student_id}, distance: {distance:.2f})")
                return student_id, name, f"Attendance marked for {name} (ID: {student_id})"
            else:
                logger.error(f"Attendance marking failed: {message}")
                return None, None, message
        else:
            logger.warning(f"No match found (distance: {distance:.2f})")
            return None, None, f"No match found (distance: {distance:.2f})"
            
    except Exception as e:
        logger.error(f"Error during attendance marking: {str(e)}")
        return None, None, f"Error: {str(e)}"