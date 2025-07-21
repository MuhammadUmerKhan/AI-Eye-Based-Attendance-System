import streamlit as st
# import numpy as np
# from src.register_students import register_student
# from src.mark_attendance import mark_attendance
# from src.utils import save_image, cleanup_temp_image
# from src.database import Database
# from src.faiss_index import FaissIndex
# from src.config import get_input_images_dir, INPUT_IMAGES_DIR, COURSES
# from src.logger import get_logger
# from src.db_setup import init_database
# from datetime import datetime
# from typing import Any
# import os

# # Configure logging
# logger = get_logger(__name__)

# Inject custom CSS
with open("./static/styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # Initialize database and FAISS index once using session state
# if 'db' not in st.session_state:
#     logger.debug("Initializing database and FAISS index")
#     success, message = init_database()
#     if not success:
#         logger.error({"message": message})
#         st.error(message)
#         st.stop()
    
#     st.session_state.db = Database()
    
#     # Initialize FAISS index
#     st.session_state.faiss_index = FaissIndex()
#     for course_id in COURSES:
#         students = st.session_state.db.fetch_students(course_id)
#         if students:
#             logger.debug(f"Building FAISS index for course {course_id} with {len(students)} students")
#             embeddings = np.array([student['embedding'] for student in students], dtype=np.float32)
#             student_ids = [student['id'] for student in students]
#             names = [student['name'] for student in students]
#             st.session_state.faiss_index.build_index(embeddings, student_ids, names, course_id)
#         else:
#             logger.debug(f"No students found for course {course_id}, loading empty FAISS index")
#             st.session_state.faiss_index.load_index(course_id)
    
#     logger.info({"message": "Initialized FAISS indices for all courses"})

# # Reference cached instances
# db = st.session_state.db
# faiss_index = st.session_state.faiss_index

st.set_page_config(page_title="AI Attendance System", layout="centered")

st.title("📸 Eye-Based Attendance System")

# Tabs
tab1, tab2 = st.tabs(["📝 Register Student", "✅ Mark Attendance"])

# Tab 1: Register Student
with tab1:
    st.subheader("📝 Register a New Student")
    
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        roll_no = st.text_input("Roll Number")
        course_id = st.selectbox("Course", options=list(["CS101", "CS102"]), format_func=lambda x: f"{x} - Sample Course") # Placeholder for COURSES
        image_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
        submit_btn = st.form_submit_button("Register")
    
    # if submit_btn:
    #     if not all([name, roll_no, course_id, image_file]):
    #         logger.error({"message": "❌ Missing required fields or image"})
    #         st.error("Please fill all fields and upload an image.")
    #     else:
    #         try:
    #             course_name = COURSES[course_id]
    #             image_path = os.path.join(get_input_images_dir(course_id), f"{roll_no}.jpg")
    #             os.makedirs(os.path.dirname(image_path), exist_ok=True)
    #             success, message = save_image(image_file, image_path)
    #             if not success:
    #                 logger.error({"message": f"❌ {message}"})
    #                 st.error(message)
    #             else:
    #                 logger.info({"roll_no": roll_no, "course_id": course_id, "message": f"🖼️ Image saved at {image_path}"})
    #                 success, message = register_student(db, roll_no, name, course_id, course_name, image_path, faiss_index)
    #                 if success:
    #                     logger.info({"roll_no": roll_no, "course_id": course_id, "message": f"✅ {message}"})
    #                     st.success(f"✅ {message}")
    #                     st.balloons()
    #                 else:
    #                     logger.error({"roll_no": roll_no, "course_id": course_id, "message": f"❌ {message}"})
    #                     st.error(f"❌ {message}")
    #             except Exception as e:
    #                 logger.error({"error": str(e), "message": "❌ Unexpected error during registration"})
    #                 st.error(f"Unexpected error: {str(e)}")

# Tab 2: Mark Attendance
with tab2:
    st.subheader("✅ Upload Photo to Mark Attendance")
    course_id = st.selectbox("Course for Attendance", options=list(["CS101", "CS102"]), format_func=lambda x: f"{x} - Sample Course", key="attendance_course") # Placeholder for COURSES
    image_to_check = st.file_uploader("Upload Group/Single Image", type=["jpg", "jpeg", "png"], key="attendance_image")
    
    # if image_to_check:
    #     try:
    #         temp_image_path = os.path.join(INPUT_IMAGES_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    #         success, message = save_image(image_to_check, temp_image_path)
    #         if not success:
    #             logger.error({"message": f"❌ {message}"})
    #             st.error(message)
    #         else:
    #             logger.info({"message": f"🖼️ Temporary image saved at {temp_image_path}"})
    #             student_id, name, message = mark_attendance(db, temp_image_path, faiss_index, course_id)
    #             if student_id:
    #                 logger.info({"message": f"✅ {message}"})
    #                 st.success(f"✅ {message}")
    #             else:
    #                 logger.warning({"message": f"❌ {message}"})
    #                 st.error(f"❌ {message}")
    #             cleanup_temp_image(temp_image_path)
    #         except Exception as e:
    #             logger.error({"error": str(e), "message": "❌ Unexpected error during attendance marking"})
    #             st.error(f"Unexpected error: {str(e)}")

# # Close database connection on app shutdown
# if st.session_state.get('shutdown', False):
#     logger.debug("Closing database connection on app shutdown")
#     db.close_connection()