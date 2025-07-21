import numpy as np
import faiss
import os
import warnings
from src.config import get_faiss_index_path, EMBEDDING_DIM
from src.logger import get_logger

# Configure logging
logger = get_logger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

class FaissIndex:
    """
    Manages course-specific persistent FAISS indices for efficient embedding matching.
    """
    def __init__(self):
        """Initialize FAISS index dictionary."""
        try:
            logger.debug("Initializing FaissIndex")
            self.indices = {}  # Dictionary to store course_id -> (index, student_ids, names)
            self.dimension = EMBEDDING_DIM
            logger.info({"message": "FaissIndex initialized successfully"})
        except Exception as e:
            logger.error({"error": str(e), "message": "Failed to initialize FaissIndex"})
            raise

    def build_index(self, embeddings: np.ndarray, student_ids: list, names: list, course_id: str):
        """
        Build and save a FAISS index for a specific course from student embeddings.
        
        Args:
            embeddings (np.ndarray): Array of student embeddings.
            student_ids (list): List of student IDs (integers).
            names (list): List of student names.
            course_id (str): Course identifier.
        """
        try:
            logger.debug(f"Building FAISS index for course {course_id} with {len(embeddings)} embeddings")
            if embeddings.size == 0:
                logger.warning({"course_id": course_id, "message": "No embeddings provided to build FAISS index"})
                return
            
            if not isinstance(embeddings, np.ndarray):
                logger.error({"type": type(embeddings), "message": "Invalid embeddings type"})
                raise ValueError("Embeddings must be a numpy array")
            
            if embeddings.shape[0] != len(student_ids) or embeddings.shape[0] != len(names):
                logger.error({"embeddings_count": embeddings.shape[0], "student_ids_count": len(student_ids), "names_count": len(names), "message": "Mismatch in lengths"})
                raise ValueError("Embeddings, student_ids, and names must have the same length")
            
            if embeddings.shape[1] != self.dimension:
                logger.error({"got": embeddings.shape[1], "expected": self.dimension, "message": f"Embedding dimension mismatch for course {course_id}"})
                raise ValueError(f"Embedding dimension must be {self.dimension}")
            
            index = faiss.IndexFlatL2(self.dimension)
            index.add(embeddings)
            self.indices[course_id] = (index, student_ids, names)
            
            index_path = get_faiss_index_path(course_id)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
            logger.info({"course_id": course_id, "message": f"FAISS index built and saved to {index_path}", "embedding_count": len(embeddings)})
        except Exception as e:
            logger.error({"error": str(e), "message": f"Failed to build FAISS index for course {course_id}"})
            raise

    def load_index(self, course_id: str):
        """
        Load FAISS index for a specific course from disk.
        
        Args:
            course_id (str): Course identifier.
        """
        try:
            index_path = get_faiss_index_path(course_id)
            logger.debug(f"Attempting to load FAISS index for course {course_id} from {index_path}")
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                if index.d != self.dimension:
                    logger.error({"got": index.d, "expected": self.dimension, "message": f"Dimension mismatch in loaded FAISS index for course {course_id}"})
                    raise ValueError(f"Loaded index dimension {index.d} does not match expected {self.dimension}")
                self.indices[course_id] = (index, [], [])  # Initialize with empty IDs and names
                logger.info({"course_id": course_id, "message": f"FAISS index loaded from {index_path}"})
            else:
                logger.warning({"course_id": course_id, "message": f"No FAISS index found at {index_path}"})
                self.indices[course_id] = (None, [], [])
        except Exception as e:
            logger.error({"error": str(e), "message": f"Failed to load FAISS index for course {course_id}"})
            self.indices[course_id] = (None, [], [])

    def update_index(self, embedding: np.ndarray, student_id: int, name: str, course_id: str):
        """
        Add a new embedding to the FAISS index for a specific course.
        
        Args:
            embedding (np.ndarray): New student embedding.
            student_id (int): Student ID.
            name (str): Student name.
            course_id (str): Course identifier.
        """
        try:
            logger.debug(f"Updating FAISS index for student_id: {student_id}, course_id: {course_id}")
            if not isinstance(embedding, np.ndarray):
                logger.error({"type": type(embedding), "message": f"Invalid embedding type for student {student_id}"})
                raise ValueError("Embedding must be a numpy array")
            
            if embedding.shape[0] != self.dimension:
                logger.error({"got": embedding.shape[0], "expected": self.dimension, "message": f"Embedding dimension mismatch for student {student_id}"})
                raise ValueError(f"Embedding dimension must match index dimension ({self.dimension})")
            
            if course_id not in self.indices or self.indices[course_id][0] is None:
                self.indices[course_id] = (faiss.IndexFlatL2(self.dimension), [], [])
                logger.debug(f"Created new FAISS index for course {course_id} with dimension {self.dimension}")
            
            index, student_ids, names = self.indices[course_id]
            index.add(np.array([embedding], dtype=np.float32))
            student_ids.append(student_id)
            names.append(name)
            self.indices[course_id] = (index, student_ids, names)
            
            index_path = get_faiss_index_path(course_id)
            faiss.write_index(index, index_path)
            logger.info({"student_id": student_id, "course_id": course_id, "message": "FAISS index updated", "embedding_count": index.ntotal})
        except Exception as e:
            logger.error({"error": str(e), "message": f"Failed to update FAISS index for student {student_id} in course {course_id}"})

    def search(self, embedding: np.ndarray, course_id: str, k: int = 1):
        """
        Search for the nearest neighbor in the FAISS index for a specific course.
        
        Args:
            embedding (np.ndarray): Query embedding.
            course_id (str): Course identifier.
            k (int): Number of nearest neighbors to return.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, list, list]: Distances, indices, student_ids, names.
        """
        try:
            logger.debug(f"Searching FAISS index for course {course_id} with k={k}")
            if course_id not in self.indices or self.indices[course_id][0] is None:
                logger.warning({"course_id": course_id, "message": "FAISS index not initialized for course"})
                return None, None, [], []
            
            index, student_ids, names = self.indices[course_id]
            if index.ntotal == 0:
                logger.warning({"course_id": course_id, "message": "FAISS index is empty for course"})
                return None, None, [], []
            
            if not isinstance(embedding, np.ndarray):
                logger.error({"type": type(embedding), "message": "Invalid embedding type"})
                raise ValueError("Embedding must be a numpy array")
            
            if embedding.shape[1] != self.dimension:
                logger.error({"got": embedding.shape[1], "expected": self.dimension, "message": "Embedding dimension mismatch"})
                raise ValueError(f"Embedding dimension must match index dimension ({self.dimension})")
            
            distances, indices = index.search(embedding, k)
            logger.info({"course_id": course_id, "message": "FAISS search completed", "distances": distances.tolist(), "indices": indices.tolist()})
            return distances, indices, student_ids, names
        except Exception as e:
            logger.error({"error": str(e), "message": f"FAISS search failed for course {course_id}"})
            return None, None, [], []