import numpy as np
import faiss
import os
import warnings
from src.config import FAISS_INDEX_PATH, EMBEDDING_DIM
from src.logger import get_logger

# Configure logging
logger = get_logger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

class FaissIndex:
    """
    Manages a persistent FAISS index for efficient embedding matching.
    """
    def __init__(self):
        """Initialize FAISS index."""
        try:
            logger.debug("Initializing FaissIndex")
            self.index = None
            self.student_ids = []
            self.names = []
            self.dimension = None
            self.load_index()
            logger.info({"message": "FaissIndex initialized successfully"})
        except Exception as e:
            logger.error({"error": str(e), "message": "Failed to initialize FaissIndex"})
            raise

    def build_index(self, embeddings: np.ndarray, student_ids: list, names: list):
        """
        Build and save a FAISS index from student embeddings.
        
        Args:
            embeddings (np.ndarray): Array of student embeddings.
            student_ids (list): List of student IDs.
            names (list): List of student names.
        """
        try:
            logger.debug(f"Building FAISS index with {len(embeddings)} embeddings")
            if embeddings.size == 0:
                logger.warning({"message": "No embeddings provided to build FAISS index"})
                return
            
            if not isinstance(embeddings, np.ndarray):
                logger.error({"type": type(embeddings), "message": "Invalid embeddings type"})
                raise ValueError("Embeddings must be a numpy array")
            
            if embeddings.shape[0] != len(student_ids) or embeddings.shape[0] != len(names):
                logger.error({"embeddings_count": embeddings.shape[0], "student_ids_count": len(student_ids), "names_count": len(names), "message": "Mismatch in lengths"})
                raise ValueError("Embeddings, student_ids, and names must have the same length")
            
            self.dimension = embeddings.shape[1]
            self.student_ids = student_ids
            self.names = names
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            
            os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            logger.info({"message": f"FAISS index built and saved to {FAISS_INDEX_PATH}", "embedding_count": len(embeddings)})
        except Exception as e:
            logger.error({"error": str(e), "message": "Failed to build FAISS index"})
            raise

    def load_index(self):
        """
        Load FAISS index from disk.
        """
        try:
            logger.debug(f"Attempting to load FAISS index from {FAISS_INDEX_PATH}")
            if os.path.exists(FAISS_INDEX_PATH):
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                self.dimension = self.index.d
                logger.info({"message": f"FAISS index loaded from {FAISS_INDEX_PATH}"})
            else:
                logger.warning({"message": f"No FAISS index found at {FAISS_INDEX_PATH}"})
                self.index = None
                self.dimension = None
        except Exception as e:
            logger.error({"error": str(e), "message": "Failed to load FAISS index"})
            self.index = None
            self.dimension = None

    def update_index(self, embedding: np.ndarray, student_id: str, name: str):
        """
        Add a new embedding to the FAISS index.
        
        Args:
            embedding (np.ndarray): New student embedding.
            student_id (str): Student ID.
            name (str): Student name.
        """
        try:
            logger.debug(f"Updating FAISS index for student_id: {student_id}")
            if not isinstance(embedding, np.ndarray):
                logger.error({"type": type(embedding), "message": f"Invalid embedding type for student {student_id}"})
                raise ValueError("Embedding must be a numpy array")
            
            if self.index is None:
                self.dimension = EMBEDDING_DIM
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.debug(f"Created new FAISS index with dimension {self.dimension}")
            
            if embedding.shape[0] != self.dimension:
                logger.error({"got": embedding.shape[0], "expected": self.dimension, "message": f"Embedding dimension mismatch for student {student_id}"})
                raise ValueError(f"Embedding dimension must match index dimension ({self.dimension})")
            
            self.index.add(np.array([embedding], dtype=np.float32))
            self.student_ids.append(student_id)
            self.names.append(name)
            
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            logger.info({"student_id": student_id, "message": "FAISS index updated", "embedding_count": self.index.ntotal})
        except Exception as e:
            logger.error({"error": str(e), "message": f"Failed to update FAISS index for student {student_id}"})

    def search(self, embedding: np.ndarray, k: int = 1):
        """
        Search for the nearest neighbor in the FAISS index.
        
        Args:
            embedding (np.ndarray): Query embedding.
            k (int): Number of nearest neighbors to return.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors.
        """
        try:
            logger.debug(f"Searching FAISS index with k={k}")
            if self.index is None:
                logger.warning({"message": "FAISS index not initialized"})
                return None, None
            
            if not isinstance(embedding, np.ndarray):
                logger.error({"type": type(embedding), "message": "Invalid embedding type"})
                raise ValueError("Embedding must be a numpy array")
            
            if self.dimension is None:
                logger.error({"message": "FAISS index dimension not set"})
                raise ValueError("FAISS index dimension must be set before search")
            
            if embedding.shape[1] != self.dimension:
                logger.error({"got": embedding.shape[1], "expected": self.dimension, "message": "Embedding dimension mismatch"})
                raise ValueError(f"Embedding dimension must match index dimension ({self.dimension})")
            
            distances, indices = self.index.search(embedding, k)
            logger.info({"message": "FAISS search completed", "distances": distances.tolist(), "indices": indices.tolist()})
            return distances, indices
        except Exception as e:
            logger.error({"error": str(e), "message": "FAISS search failed"})
            return None, None