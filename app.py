import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import numpy as np
import cv2
import tempfile
import os
import json
from glob import glob
from collections import Counter
from detect import run 
import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

st.set_page_config(page_title="YOLOv5 Skin Burn Detection", page_icon="🔥")

# --- Fungsi untuk memuat CSS eksternal ---
def load_css(file_name):
    """Loads a CSS file and applies it using st.markdown."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Styles might not be applied.")

# --------- Load navbar ---------
pages = ["Beranda", "Deteksi"]
# selected_page = st_navbar(pages, selected="HOME")

# Initialize state satu kali aja
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Beranda"

# Navbar DIJALANKAN tapi jangan langsung set ke session_state!
navbar_choice = st_navbar(pages, selected=st.session_state.selected_page)

# Kalau user klik navbar, baru update state
if navbar_choice != st.session_state.selected_page:
    st.session_state.selected_page = navbar_choice
    st.rerun()





# --------- Load penanganan.json ---------
@st.cache_data
def load_penanganan(path="penanganan.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return {item["name"]: item["instructions"] for item in data}

penanganan_dict = load_penanganan()
class_names = ['First-Degree', 'Forth-Degree', 'Second-Degree', 'Third-Degree']

# --------- detection functions ---------
def load_image_opencv(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def get_latest_exp_folder(base_path="runs/detect"):
    exp_folders = sorted(glob(os.path.join(base_path, "exp*")), key=os.path.getmtime)
    return exp_folders[-1] if exp_folders else None

def read_detection_labels(txt_path):
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            labels = [int(line.split()[0]) for line in f.readlines()]
    return labels

def run_detection(file):
    image = load_image_opencv(file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    result = {"filename": file.name, "labels": [], "image_path": None}

    try:
        run(weights='./best_old.pt', source=tmp_path, conf_thres=0.3, imgsz=(640, 640), save_txt=True, save_conf=True, save_crop=False)
        latest_exp = get_latest_exp_folder()

        if latest_exp:
            img_path = os.path.join(latest_exp, os.path.basename(tmp_path))
            txt_path = os.path.join(latest_exp, "labels", os.path.splitext(os.path.basename(tmp_path))[0] + ".txt")
            labels = read_detection_labels(txt_path)
            label_names = [class_names[c] for c in labels]

            result.update({"labels": label_names, "image_path": img_path})
        else:
            st.warning("❌ Folder hasil deteksi tidak ditemukan.")
    except Exception as e:
        st.error(f"⚠️ Deteksi gagal untuk {file.name}: {e}")
    finally:
        os.remove(tmp_path)

    return result




if st.session_state['selected_page'] == "Beranda":
    load_css("style.css")
    st.markdown('<div class="hero">\
        <h1>Deteksi Luka Bakar</h1>\
        <p>Aplikasi ini membantu Anda mendeteksi tingkat luka bakar kulit menggunakan YOLOv5.</p>\
        <p>Unggah gambar di halaman <b>Deteksi</b> untuk memulai.</p>\
        </div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Informasi Derajat Luka Bakar")
    # Derajat 1
    with st.expander("Luka Bakar Derajat 1"):
        st.write("Luka bakar derajat 1 hanya mengenai lapisan epidermis, menyebabkan kemerahan dan nyeri tanpa luka terbuka.")

    # Derajat 2
    with st.expander("Luka Bakar Derajat 2"):
        st.write("Luka bakar derajat 2 melibatkan lapisan epidermis dan dermis, dengan lepuhan dan nyeri yang lebih intens.")

    # Derajat 3
    with st.expander("Luka Bakar Derajat 3"):
        st.write("Luka bakar derajat 3 merusak seluruh lapisan kulit, menyebabkan jaringan mati dan memerlukan penanganan medis serius.")

    # Derajat 4
    with st.expander("Luka Bakar Derajat 4"):
        st.write("Luka bakar derajat 4 mencapai otot dan tulang, merupakan kondisi yang sangat serius dan memerlukan perawatan intensif.")
        
        
    st.markdown("---")
    
elif st.session_state['selected_page'] == "Deteksi":
    load_css("style.css")
    st.markdown('<div class="hero">\
    <h1>Deteksi Luka Bakar</h1>\
    <p>Aplikasi ini membantu Anda mendeteksi tingkat luka bakar kulit menggunakan YOLOv5.</p>\
    <p>Upload gambar kulitmu dan <b>Deteksi</b> tingkat luka bakarnya secara otomatis.</p>\
    </div>', unsafe_allow_html=True)
    st.markdown("")

    if 'detection_results' not in st.session_state:
        st.session_state['detection_results'] = []

    uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"], key="uploaded_file")

    col1, col2 = st.columns(2)
    with col1:
        detect_btn = st.button("🚀 Jalankan Deteksi")
    with col2:
        clear_btn = st.button("🔄 Clear Semua")

    if detect_btn:
        if uploaded_file is None:
            st.warning("⚠️ Harap upload gambar terlebih dahulu.")
        else:
            st.session_state['detection_results'] = []  # Reset
            with st.spinner("Sedang mendeteksi..."):
                result = run_detection(uploaded_file)
                st.session_state['detection_results'].append(result)

    if clear_btn:
        keys_to_clear = ['detection_results', 'uploaded_file']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.session_state['detection_results']:
        for result in st.session_state['detection_results']:
            st.markdown(f"### Hasil Deteksi - {result['filename']}")
            if result['image_path'] and os.path.exists(result["image_path"]):
                st.image(result["image_path"], use_container_width=True)

            if result['labels']:
                counts = Counter(result["labels"])
                st.markdown("**Kelas yang Terdeteksi:**")
                for label, count in counts.items():
                    st.write(label)

                st.header("Penanganan yang Disarankan:")
                for label in set(result["labels"]):
                    st.subheader(f"{label}")
                    for i, step in enumerate(penanganan_dict.get(label, ["Tidak ada informasi penanganan."]), 1):
                        st.markdown(f"{i}. {step}")
            else:
                st.info("Tidak ada luka bakar terdeteksi.")
    else:
        st.info("Silakan upload gambar dan klik tombol 'Jalankan Deteksi' untuk memulai.")
else:
    st.error("Page tidak ditemukan. Kembali ke Home.")
