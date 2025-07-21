import sqlite3
import os
import warnings
import numpy as np
from typing import Optional, Tuple, List, Dict
from src.config import DATABASE_PATH, EMBEDDING_DIM
from src.logger import get_logger

# Configure logging
logger = get_logger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

class Database:
    """
    Attendance System Database Operations 🗄️

    Handles all SQLite interactions for the eye-based attendance system.
    """
    def __init__(self, db_path: str = DATABASE_PATH):
        """🗄️ Initialize and connect to SQLite database."""
        try:
            logger.debug(f"Connecting to database at {db_path}")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            self.init_tables()
            logger.info({"message": f"Connected to SQLite database at {db_path}"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Database connection failed"})
            raise

    def init_tables(self) -> None:
        """📋 Initialize students and attendance tables."""
        try:
            logger.debug("Initializing database tables")
            with self.connection:
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS students (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        department TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                """)
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (student_id) REFERENCES students (id)
                    )
                """)
            logger.info({"message": "Database tables initialized"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Failed to initialize tables"})
            raise

    def register_student(self, student_id: str, name: str, department: str, embedding: np.ndarray) -> Tuple[bool, str]:
        """📝 Register a student in the database."""
        try:
            logger.debug(f"Registering student {student_id}, name: {name}, department: {department}, embedding shape: {embedding.shape}")
            if not isinstance(embedding, np.ndarray):
                logger.error({"type": type(embedding), "message": f"Invalid embedding type for student {student_id}"})
                raise ValueError("Embedding must be a numpy array")
            
            embedding = embedding.astype(np.float32)
            if embedding.shape[0] != EMBEDDING_DIM:
                logger.error({"got": embedding.shape[0], "expected": EMBEDDING_DIM, "message": f"Unexpected embedding dimension for student {student_id}"})
                raise ValueError(f"Embedding dimension must be {EMBEDDING_DIM}, got {embedding.shape[0]}")
            
            embedding_bytes = embedding.tobytes()
            expected_bytes = EMBEDDING_DIM * 4
            if len(embedding_bytes) != expected_bytes:
                logger.error({"got": len(embedding_bytes), "expected": expected_bytes, "message": f"Unexpected embedding bytes length for student {student_id}"})
                raise ValueError(f"Embedding bytes length must be {expected_bytes}, got {len(embedding_bytes)}")
            
            with self.connection:
                self.cursor.execute(
                    "INSERT OR REPLACE INTO students (id, name, department, embedding) VALUES (?, ?, ?, ?)",
                    (student_id, name, department, embedding_bytes)
                )
            logger.info({"student_id": student_id, "message": f"Student {name} registered successfully", "embedding_shape": embedding.shape})
            return True, f"Student {name} (ID: {student_id}) registered successfully"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Failed to register student"})
            return False, f"Database error: {str(e)}"
        except Exception as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Unexpected error during student registration"})
            return False, f"Error: {str(e)}"

    def fetch_students(self) -> Optional[List[Dict]]:
        """📋 Fetch all students' data."""
        try:
            logger.debug("Fetching all students from database")
            with self.connection:
                self.cursor.execute("SELECT id, name, department, embedding FROM students")
                students = [dict(row) for row in self.cursor.fetchall()]
            for student in students:
                embedding_bytes = student['embedding']
                expected_bytes = EMBEDDING_DIM * 4
                if len(embedding_bytes) != expected_bytes:
                    logger.error({"student_id": student['id'], "got": len(embedding_bytes), "expected": expected_bytes, "message": "Unexpected embedding bytes length in database"})
                    raise ValueError(f"Unexpected embedding bytes length {len(embedding_bytes)} for student {student['id']}, expected {expected_bytes}")
                
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                if embedding.shape[0] != EMBEDDING_DIM:
                    logger.error({"student_id": student['id'], "got": embedding.shape[0], "expected": EMBEDDING_DIM, "message": "Unexpected embedding dimension in database"})
                    raise ValueError(f"Unexpected embedding dimension {embedding.shape[0]} for student {student['id']}, expected {EMBEDDING_DIM}")
                
                student['embedding'] = embedding
            logger.info({"count": len(students), "message": "Fetched student data"})
            return students or None
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Error fetching students"})
            return None
        except Exception as e:
            logger.error({"error": str(e), "message": "Unexpected error fetching students"})
            return None

    def mark_attendance(self, student_id: str, timestamp: str) -> Tuple[bool, str]:
        """✅ Mark attendance for a student."""
        try:
            logger.debug(f"Marking attendance for student_id: {student_id}, timestamp: {timestamp}")
            with self.connection:
                self.cursor.execute(
                    "INSERT INTO attendance (student_id, timestamp) VALUES (?, ?)",
                    (student_id, timestamp)
                )
            logger.info({"student_id": student_id, "message": "Attendance marked"})
            return True, f"Attendance marked for student {student_id}"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Error marking attendance"})
            return False, f"Error marking attendance: {str(e)}"
        except Exception as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Unexpected error marking attendance"})
            return False, f"Error: {str(e)}"

    def close_connection(self):
        """🔒 Close SQLite connection."""
        try:
            logger.debug("Closing database connection")
            self.connection.close()
            logger.info({"message": "Database connection closed"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Error closing connection"})
        except Exception as e:
            logger.error({"error": str(e), "message": "Unexpected error closing connection"})