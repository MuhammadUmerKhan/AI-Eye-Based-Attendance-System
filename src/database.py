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
                        id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        course_id TEXT NOT NULL,
                        course_name TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        PRIMARY KEY (id, course_id)
                    )
                """)
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        course_id TEXT NOT NULL,
                        FOREIGN KEY (student_id, course_id) REFERENCES students (id, course_id)
                    )
                """)
            logger.info({"message": "Database tables initialized"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Failed to initialize tables"})
            raise

    def check_duplicate(self, student_id: int, course_id: str) -> bool:
        """Check if a student is already registered for a course."""
        try:
            logger.debug(f"Checking for duplicate student {student_id} in course {course_id}")
            with self.connection:
                self.cursor.execute(
                    "SELECT 1 FROM students WHERE id = ? AND course_id = ?",
                    (student_id, course_id)
                )
                exists = self.cursor.fetchone() is not None
            logger.debug({"student_id": student_id, "course_id": course_id, "exists": exists, "message": "Duplicate check completed"})
            return exists
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": f"Error checking duplicate for student {student_id} in course {course_id}"})
            return False

    def register_student(self, student_id: int, name: str, course_id: str, course_name: str, embedding: np.ndarray) -> Tuple[bool, str]:
        """📝 Register a student in the database."""
        try:
            logger.debug(f"Registering student {student_id}, name: {name}, course_id: {course_id}")
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
                    "INSERT OR REPLACE INTO students (id, name, course_id, course_name, embedding) VALUES (?, ?, ?, ?, ?)",
                    (student_id, name, course_id, course_name, embedding_bytes)
                )
            logger.info({"student_id": student_id, "course_id": course_id, "message": f"Student {name} registered successfully"})
            return True, f"Student {name} (ID: {student_id}) registered successfully for {course_name}"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "course_id": course_id, "error": str(e), "message": "Failed to register student"})
            return False, f"Database error: {str(e)}"
        except Exception as e:
            logger.error({"student_id": student_id, "course_id": course_id, "error": str(e), "message": "Unexpected error during student registration"})
            return False, f"Error: {str(e)}"

    def fetch_students(self, course_id: Optional[str] = None) -> Optional[List[Dict]]:
        """📋 Fetch all students' data, optionally filtered by course_id."""
        try:
            logger.debug(f"Fetching students from database, course_id: {course_id if course_id else 'all'}")
            with self.connection:
                if course_id:
                    self.cursor.execute("SELECT id, name, course_id, course_name, embedding FROM students WHERE course_id = ?", (course_id,))
                else:
                    self.cursor.execute("SELECT id, name, course_id, course_name, embedding FROM students")
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
            logger.info({"count": len(students), "course_id": course_id if course_id else "all", "message": "Fetched student data"})
            return students or None
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Error fetching students"})
            return None
        except Exception as e:
            logger.error({"error": str(e), "message": "Unexpected error fetching students"})
            return None

    def mark_attendance(self, student_id: int, timestamp: str, course_id: str) -> Tuple[bool, str]:
        """✅ Mark attendance for a student."""
        try:
            logger.debug(f"Marking attendance for student_id: {student_id}, course_id: {course_id}, timestamp: {timestamp}")
            with self.connection:
                self.cursor.execute(
                    "INSERT INTO attendance (student_id, course_id, timestamp) VALUES (?, ?, ?)",
                    (student_id, course_id, timestamp)
                )
            logger.info({"student_id": student_id, "course_id": course_id, "message": "Attendance marked"})
            return True, f"Attendance marked for student {student_id} in course {course_id}"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "course_id": course_id, "error": str(e), "message": "Error marking attendance"})
            return False, f"Error marking attendance: {str(e)}"
        except Exception as e:
            logger.error({"student_id": student_id, "course_id": course_id, "error": str(e), "message": "Unexpected error marking attendance"})
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