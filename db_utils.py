import sqlite3
from datetime import datetime

# ✅ Connect to DB (or create if not exists)
def get_connection():
    return sqlite3.connect('attendance.db')


# ✅ Create tables
def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    # Students table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_no TEXT UNIQUE NOT NULL,
        department TEXT,
        image_path TEXT,
        embedding_path TEXT,
        registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Attendance table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        attendance_date DATE,
        attendance_time TIME,
        confidence REAL,
        marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    )
    ''')

    conn.commit()
    conn.close()


# ✅ Register student
def register_student(name, roll_no, department, image_path, embedding_path):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO students (name, roll_no, department, image_path, embedding_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, roll_no, department, image_path, embedding_path))
        conn.commit()
        student_id = cur.lastrowid
        print(f"Student registered with ID: {student_id}")
        return student_id
    except Exception as e:
        print("Error registering student:", e)
        return None
    finally:
        conn.close()


# ✅ Mark attendance
from datetime import datetime

def mark_attendance(student_id, confidence):
    now = datetime.now()
    conn = sqlite3.connect('attendance.db')
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO attendance (student_id, attendance_date, attendance_time, confidence)
            VALUES (?, ?, ?, ?)
        ''', (
            student_id,
            now.date().isoformat(),       # 'YYYY-MM-DD'
            now.time().strftime("%H:%M:%S"),  # 'HH:MM:SS'
            confidence
        ))
        conn.commit()
        print(f"Attendance marked for student ID {student_id}")
    except Exception as e:
        print("Error marking attendance:", e)
    finally:
        conn.close()
