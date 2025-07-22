# 📸 AI Eye-Based Attendance System 🚀

![](https://faceitsystems.com/wp-content/uploads/2024/06/5597099_56379-420x300.jpg)

Welcome to the **AI Eye-Based Attendance System**, a cutting-edge solution for automated attendance tracking using facial recognition with a focus on eye region embeddings. This project leverages advanced computer vision and machine learning to register students, store their embeddings, and mark attendance efficiently. Built with Python, it integrates `DeepFace`, `MediaPipe`, `FAISS`, and `Streamlit` for a robust and user-friendly experience. 🌟

## 🎯 Project Overview

This system automates attendance tracking by:
- **Extracting Eye Region Embeddings**: Uses `MediaPipe` to detect eye regions and `DeepFace` (ArcFace model) to generate 512-dimensional embeddings.
- **Storing Data**: Saves student details and embeddings in a SQLite database and builds a FAISS index for fast similarity search.
- **Marking Attendance**: Matches new images against stored embeddings to record attendance with a timestamp.
- **Web Interface**: Provides a `Streamlit` UI for easy student registration and attendance marking.

The codebase is optimized for performance with a singleton database connection, cached `MediaPipe FaceMesh`, and log rotation, ensuring scalability and reliability. 📊

## ✨ Features

- **📝 Student Registration**: Register students with ID, name, department, and a face image. Duplicate IDs are prevented with validation checks.
- **✅ Attendance Marking**: Upload images to match against stored embeddings and record attendance with timestamps.
- **🔍 Fast Matching**: Uses FAISS `IndexFlatL2` for efficient nearest-neighbor search of embeddings.
- **🗄️ Persistent Storage**: Stores student data and embeddings in a SQLite database with an index on `students.id` for fast lookups.
- **🖼️ Image Processing**: Extracts eye regions using `MediaPipe FaceMesh` and preprocesses them for consistent embeddings.
- **📜 Logging**: Detailed logs with emojis (🚀, ✅, ❌) in `../logs/attendance.log` with rotation to manage file size.
- **🌐 Web Interface**: Streamlit app for user-friendly registration and attendance marking.

## 🛠️ Requirements

To run the project, ensure you have the following installed:

- Python 3.12.8 🐍
- Dependencies (listed in `requirements.txt`):
  - `deepface==0.0.93`
  - `tf_keras==2.15.0`
  - `opencv-python`
  - `mediapipe`
  - `faiss-cpu`
  - `streamlit`
  - `numpy`
  - `sqlite3` (built-in with Python)

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ai-eye-based-attendance-system.git
   cd ai-eye-based-attendance-system
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create Directory Structure**:
   ```bash
   mkdir -p ../database ../logs ../images/train_imgs ../images/input_imgs
   ```

5. **Add Sample Image**:
   - Place a sample face image (e.g., `123.jpg`) in `../images/train_imgs/` for testing.

## 🚀 Usage

### 1. Run the Main Script
To test the core functionality (register a student, build FAISS index, and mark attendance):
```bash
python main.py
```

**Expected Output**:
```
=== Registered Students ===
ID: 123, Name: John Doe, Department: Computer Science, Embedding: [length: 512]
Total students: 1
Attendance marked for John Doe (ID: 123)
```

**Logs**: Check `../logs/attendance.log` for detailed logs with emojis:
```
2025-07-22 14:59:00,123 - INFO - {"message": "🗄️ Connected to SQLite database at ../database/attendance.db"}
2025-07-22 14:59:00,124 - INFO - {"message": "✅ Student John Doe registered", "embedding_shape": [512]}
2025-07-22 14:59:00,125 - INFO - {"message": "✅ Attendance marked for John Doe (ID: 123)"}
```

### 2. Run the Streamlit App
To use the web interface:
```bash
streamlit run app.py
```

- **Register Student**:
  - Navigate to the "Register Student" tab.
  - Enter name, roll number, department, and upload a face image.
  - Click "Register". If the roll number already exists, an error (`❌ Student ID already registered`) will appear.
- **Mark Attendance**:
  - Navigate to the "Mark Attendance" tab.
  - Upload a face image to match against registered students.
  - View the result (e.g., `✅ Attendance marked for John Doe (ID: 123)`).

### 3. Test in Jupyter Notebook
To test embedding extraction:
```bash
jupyter notebook notebook.ipynb
```
Run the cells to verify `DeepFace` embedding extraction. If you encounter a `'KerasHistory' object has no attribute 'layer'` error, reinstall dependencies:
```bash
pip install deepface==0.0.93 tf_keras==2.15.0
```

## 🐛 Troubleshooting

- **Embedding Dimension Mismatch**:
  - Ensure `deepface==0.0.93` outputs 512-dimensional embeddings.
  - Clear `../database/attendance.db` and `../database/faiss_index.bin`:
    ```bash
    rm ../database/attendance.db ../database/faiss_index.bin
    ```
- **Image Loading Errors**:
  - Verify images in `../images/train_imgs/` are valid and contain clear faces.
- **FAISS Index Issues**:
  - If `FAISS search failed`, rebuild the index by running `main.py`.
- **Logs**: Check `../logs/attendance.log` for detailed error messages with emojis (e.g., ❌ for failures).

## 📚 Project Structure

```
ai-eye-based-attendance-system/
├── src/
│   ├── database.py         🗄️ SQLite database operations
│   ├── db_setup.py        📋 Database initialization
│   ├── extract_embeddings.py 👁️ Eye region extraction and embedding
│   ├── register_students.py 📝 Student registration
│   ├── mark_attendance.py  ✅ Attendance marking
│   ├── faiss_index.py     🔍 FAISS index management
│   ├── logger.py          📜 Logging configuration
│   ├── config.py          ⚙️ Configuration settings
│   ├── utils.py           🛠️ Utility functions
├── app.py                 🌐 Streamlit web interface
├── notebook.ipynb         📓 Jupyter notebook for testing
├── requirements.txt       📋 Python dependencies
├── ../database/           🗄️ SQLite database and FAISS index
├── ../logs/              📜 Log files
├── ../images/            🖼️ Image directories
│   ├── train_imgs/       📸 Training images
│   ├── input_imgs/       📸 Input images for attendance
```

## 👥 Contributors

- **[Zuhair Khan](https://www.linkedin.com/in/hafiz-muhammad-zuhair-khan-072347290/)**: Lead developer, implemented core logic and database integration. 🧑‍💻
- **[Muhammad Ahsan Siddiqui](https://www.linkedin.com/in/muhammad-ahsan-siddiqui-951029321/)**: Designed the Streamlit UI and optimized image processing. 🎨
- **[Muhammad Umer Khan]([https://www.linkedin.com/in/muhammad-umer-khan/](https://www.linkedin.com/in/muhammad-umer-khan-61729b260/))**: Enhanced FAISS index performance and logging system. 🚀

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- `DeepFace` for facial recognition embeddings.
- `MediaPipe` for eye region detection.
- `FAISS` for efficient similarity search.
- `Streamlit` for the web interface.

Happy attendance tracking! 🎉
