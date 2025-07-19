import sqlite3
from typing import Optional, Tuple, List, Dict
import numpy as np
from config import DATABASE_PATH
from logger import get_logger
import os

logger = get_logger(__name__)

class Database():
    """
    Attendance System Database Operations 🗄️

    Handles all SQLite interactions for the eye-based attendance system.
    """
    def __init__(self, db_path: str = DATABASE_PATH):
        """🗄️ Initialize and connect to SQLite database."""
        try:
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
            self.connection.commit()
            logger.info({"message": "Database tables initialized"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Failed to initialize tables"})
            raise

    def register_student(self, student_id: str, name: str, department: str, embedding: np.ndarray) -> Tuple[bool, str]:
        """📝 Register a student in the database.

        Args:
            student_id (str): Student ID.
            name (str): Student name.
            department (str): Department.
            embedding (np.ndarray): Eye region embedding.

        Returns:
            Tuple[bool, str]: Success status and message.
        """
        try:
            self.cursor.execute(
                "INSERT OR REPLACE INTO students (id, name, department, embedding) VALUES (?, ?, ?, ?)",
                (student_id, name, department, embedding.tobytes())
            )
            self.connection.commit()
            logger.info({"student_id": student_id, "message": f"Student {name} registered successfully"})
            return True, f"Student {name} (ID: {student_id}) registered successfully"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Failed to register student"})
            return False, f"Database error: {str(e)}"

    def fetch_students(self) -> Optional[List[Dict]]:
        """📋 Fetch all students' data.

        Returns:
            Optional[List[Dict]]: List of student dictionaries or None if error.
        """
        try:
            self.cursor.execute("SELECT * FROM students")
            students = [dict(row) for row in self.cursor.fetchall()]
            logger.info({"count": len(students), "message": "Fetched student data"})
            return students or None
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Error fetching students"})
            return None

    def mark_attendance(self, student_id: str, timestamp: str) -> Tuple[bool, str]:
        """✅ Mark attendance for a student.

        Args:
            student_id (str): Student ID.
            timestamp (str): Timestamp of attendance.

        Returns:
            Tuple[bool, str]: Success status and message.
        """
        try:
            self.cursor.execute(
                "INSERT INTO attendance (student_id, timestamp) VALUES (?, ?)",
                (student_id, timestamp)
            )
            self.connection.commit()
            logger.info({"student_id": student_id, "message": "Attendance marked"})
            return True, f"Attendance marked for student {student_id}"
        except sqlite3.Error as e:
            logger.error({"student_id": student_id, "error": str(e), "message": "Error marking attendance"})
            return False, f"Error marking attendance: {str(e)}"

    def close_connection(self):
        """🔒 Close SQLite connection."""
        try:
            self.connection.close()
            logger.info({"message": "Database connection closed"})
        except sqlite3.Error as e:
            logger.error({"error": str(e), "message": "Error closing connection"})