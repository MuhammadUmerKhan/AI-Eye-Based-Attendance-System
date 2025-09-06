import streamlit as st
import numpy as np
from src.register_students import register_student
from src.mark_attendance import mark_attendance
from src.utils import save_image, cleanup_temp_image
from src.database import Database
from src.faiss_index import FaissIndex
from src.config import get_input_images_dir, INPUT_IMAGES_DIR, COURSES, STATIC_PATH
from src.logger import get_logger
from src.db_setup import init_database
from datetime import datetime
import os

# ---------- Page Setup ----------
st.set_page_config(page_title="AI Attendance System", layout="wide", initial_sidebar_state="expanded", page_icon="🙋🏻‍♂️")


# Inject custom CSS
with open(STATIC_PATH, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<div class='title'>📸 AI Attendance System</div>", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<h3 style='color: #ff6f61;'>📚 About the Project</h3>", unsafe_allow_html=True)
    st.markdown("The AI Attendance System leverages facial recognition technology to streamline student registration and attendance tracking. Built with advanced AI, it ensures accurate and efficient management of course attendance.")

# ---------- Initialize Logger, DB, FAISS ----------
logger = get_logger(__name__)

if 'db' not in st.session_state:
    success, message = init_database()
    if not success:
        st.error(message)
        st.stop()
    st.session_state.db = Database()
    st.session_state.faiss_index = FaissIndex()
    for course_id in COURSES:
        students = st.session_state.db.fetch_students(course_id)
        if students:
            embeddings = np.array([student['embedding'] for student in students], dtype=np.float32)
            student_ids = [student['id'] for student in students]
            names = [student['name'] for student in students]
            st.session_state.faiss_index.build_index(embeddings, student_ids, names, course_id)
        else:
            st.session_state.faiss_index.load_index(course_id)

# Cached instances
db = st.session_state.db
faiss_index = st.session_state.faiss_index

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📝 Register Student", "✅ Mark Attendance"])

# ---------- Tab 1: Register Student ----------
with tab1:
    st.markdown("### 📝 Register a New Student", unsafe_allow_html=True)
    st.markdown("Add a new student to the system with their details and face image.")

    with st.form("registration_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Full Name", placeholder="Enter student's full name")
        with col2:
            roll_no = st.text_input("Roll Number", placeholder="Enter 4-digit roll number (e.g., 1234)")
        with col3:
            course_id = st.selectbox("Course", options=list(COURSES.keys()), format_func=lambda x: f"{x} - {COURSES[x]}")

        image_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"], help="Upload a clear face image (jpg, jpeg, or png).")

        submit_btn = st.form_submit_button("Register Student")

    if submit_btn:
        # Validate roll number
        try:
            roll_no_int = int(roll_no)
            if not (1000 <= roll_no_int <= 9999):
                st.error("Roll number must be exactly 4 digits (e.g., 1234).")
            elif not all([name, roll_no, course_id, image_file]):
                st.error("Please fill all fields and upload an image.")
            else:
                try:
                    image_path = os.path.join(get_input_images_dir(course_id), f"{roll_no_int}.jpg")
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    success, message = save_image(image_file, image_path)
                    if not success:
                        st.error(message)
                    else:
                        course_name = COURSES[course_id]
                        success, message = register_student(db, roll_no_int, name, course_id, course_name, image_path, faiss_index)
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                        else:
                            st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
        except ValueError:
            st.error("Roll number must be a valid integer (e.g., 1234).")

# ---------- Tab 2: Mark Attendance ----------
with tab2:
    st.markdown("### ✅ Mark Attendance", unsafe_allow_html=True)
    st.markdown("Upload a group or single image to mark attendance for a course.")

    course_id = st.selectbox("Course for Attendance", options=list(COURSES.keys()), format_func=lambda x: f"{x} - {COURSES[x]}", key="attendance_course")
    image_to_check = st.file_uploader("Upload Group/Single Image", type=["jpg", "jpeg", "png"], key="attendance_image", help="Upload an image containing one or more faces.")

    if image_to_check:
        try:
            temp_image_path = os.path.join(INPUT_IMAGES_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            success, message = save_image(image_to_check, temp_image_path)
            if not success:
                st.error(message)
            else:
                student_id, name, message = mark_attendance(db, temp_image_path, faiss_index, course_id)
                if student_id:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                cleanup_temp_image(temp_image_path)
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# ---------- Cleanup ----------
if st.session_state.get('shutdown', False):
    db.close_connection()