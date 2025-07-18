from db_utils import create_tables, register_student, mark_attendance

# Create tables on first run
create_tables()

# Register a test student
sid = register_student(
    name="Ali",
    roll_no="BSC-124",  # ← change to a new unique roll number
    department="CS",
    image_path="images/ali.jpg",
    embedding_path="embeddings/ali.npy"
)


# Mark attendance for that student
if sid:
    mark_attendance(sid, confidence=0.95)
