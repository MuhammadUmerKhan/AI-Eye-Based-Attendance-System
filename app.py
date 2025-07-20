import streamlit as st
from src.register_students import register_student
from src.mark_attendance import mark_attendance
from src.utils import save_image, cleanup_temp_image
from src.database import Database
from src.faiss_index import FaissIndex
from src.config import TRAIN_IMAGES_DIR, INPUT_IMAGES_DIR
from src.logger import get_logger
from datetime import datetime
import os
from typing import Any

# Configure logging
logger = get_logger(__name__)

# Initialize database
db = Database()

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
        department = st.text_input("Department")
        image_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
        submit_btn = st.form_submit_button("Register")
    
    if submit_btn:
        if not all([name, roll_no, department, image_file]):
            logger.error({"message": "❌ Missing required fields or image"})
            st.error("Please fill all fields and upload an image.")
        else:
            try:
                image_path = os.path.join(TRAIN_IMAGES_DIR, f"{roll_no}.jpg")
                success, message = save_image(image_file, image_path)
                if not success:
                    logger.error({"message": f"❌ {message}"})
                    st.error(message)
                else:
                    logger.info({"roll_no": roll_no, "message": f"🖼️ Image saved at {image_path}"})
                    success, message = register_student(db, roll_no, name, department, image_file)
                    if success:
                        logger.info({"roll_no": roll_no, "message": f"✅ {message}"})
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        logger.error({"roll_no": roll_no, "message": f"❌ {message}"})
                        st.error(f"❌ {message}")
            except Exception as e:
                logger.error({"error": str(e), "message": "❌ Unexpected error during registration"})
                st.error(f"Unexpected error: {str(e)}")

# Tab 2: Mark Attendance
with tab2:
    st.subheader("✅ Upload Photo to Mark Attendance")
    
    image_to_check = st.file_uploader("Upload Group/Single Image", type=["jpg", "jpeg", "png"], key="attendance_image")
    
    if image_to_check:
        try:
            temp_image_path = os.path.join(INPUT_IMAGES_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            success, message = save_image(image_to_check, temp_image_path)
            if not success:
                logger.error({"message": f"❌ {message}"})
                st.error(message)
            else:
                logger.info({"message": f"🖼️ Temporary image saved at {temp_image_path}"})
                student_id, name, message = mark_attendance(db, temp_image_path, FaissIndex())
                if student_id:
                    logger.info({"message": f"✅ {message}"})
                    st.success(f"✅ {message}")
                else:
                    logger.warning({"message": f"❌ {message}"})
                    st.error(f"❌ {message}")
                cleanup_temp_image(temp_image_path)
        except Exception as e:
            logger.error({"error": str(e), "message": "❌ Unexpected error during attendance marking"})
            st.error(f"Unexpected error: {str(e)}")

# Close database connection on app shutdown
if st.session_state.get('shutdown', False):
    db.close_connection()