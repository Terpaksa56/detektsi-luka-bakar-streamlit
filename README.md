# YOLOv5 Skin Burn Detection

This is a web application that uses YOLOv5 to detect the degree of skin burns from an uploaded image. The application is built with Streamlit and is designed to be easy to use.

## How to Run the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Terpaksa56/deteksi-luka-bakar-streamlit.git
   cd deteksi-luka-bakar-streamlit
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Technologies Used

- **YOLOv5:** For object detection to identify the degree of skin burns.
- **Streamlit:** For creating the web application.
- **OpenCV:** For image processing.
- **PyTorch:** As the backend for YOLOv5.

## File Descriptions

- `app.py`: The main Streamlit application file.
- `detect.py`: Contains the YOLOv5 detection logic.
- `best_old.pt`: The trained YOLOv5 model weights.
- `requirements.txt`: A list of the Python dependencies.
- `penanganan.json`: Contains information about how to treat different degrees of burns.
- `models/`: Contains the YOLOv5 model architecture.
- `utils/`: Contains utility functions for the YOLOv5 model.
