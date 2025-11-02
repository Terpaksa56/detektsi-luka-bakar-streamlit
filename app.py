import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import streamlit.components.v1 as components

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


# Set page config with new branding
st.set_page_config(page_title="TERLUKA - Deteksi Luka Bakar", page_icon="logo-terluka.png", layout="wide")
modern_css = """
<style>
    /* TERLUKA Branding Title & Logo */
    .terluka-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
        margin-top: 0.5rem;
    }
    .terluka-logo {
        height: 56px;
        width: 56px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(220,38,38,0.12);
        background: #fff;
        object-fit: contain;
        border: 2px solid #dc2626;
    }
    .terluka-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #dc2626;
        text-shadow: 0 2px 8px rgba(220,38,38,0.08);
        font-family: 'Inter', sans-serif;
        margin-bottom: 0;
        margin-top: 0;
    }
    @media (prefers-color-scheme: dark) {
        .terluka-header {
            background: none;
        }
        .terluka-logo {
            background: #232326;
            border: 2px solid #991b1b;
        }
        .terluka-title {
            color: #f87171;
            text-shadow: 0 2px 8px rgba(220,38,38,0.18);
        }
    }
    /* Border tegas untuk expander (override Streamlit) */
    div[data-testid="stExpander"] {
        border: 1px solid #222 !important;
        border-radius: 14px !important;
        box-sizing: border-box;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stExpander"] > div:first-child {
        border-bottom: 1.5px solid #222 !important;
        border-radius: 14px 14px 0 0 !important;
    }
    /* Hero subtitle khusus: putih di hero-section */
    .hero-section .hero-subtitle { color: #fff !important; }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    /* GLOBAL: Semua font hitam */
    html, body, .stApp, * {
        color: #111 !important;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    /* Header Styling */
    h1, h2, h3, h4, h5, h6, p, span, strong, ul, li, .hero-section, .hero-title, .hero-subtitle, .badge, .info-card, .feature-item, .streamlit-expanderHeader, .streamlit-expanderContent, .result-card, .instruction-step {
        color: #111 !important;
    }
    h1 {
        font-size: 3rem !important;
        margin-bottom: 1rem !important;
    }
    /* Card Styling */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #dc2626;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(220, 38, 38, 0.3);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: #fff !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.3);
        width: 100%;
        text-shadow: none !important;
    }
    .stButton > button * {
        color: #fff !important;
        text-shadow: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(220, 38, 38, 0.4);
        color: #fff !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        border-left: 4px solid #dc2626;
        font-weight: 600;
        color: #1a1a1a !important;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #fef2f2;
        border-left-color: #991b1b;
    }
    
    /* Expander Content */
    .streamlit-expanderContent {
        color: #1a1a1a !important;
    }
    
    .streamlit-expanderContent p,
    .streamlit-expanderContent li,
    .streamlit-expanderContent h4,
    .streamlit-expanderContent strong {
        color: #1a1a1a !important;
    }
    
    /* Upload Section */
    .uploadedFile {
        background: white;
        border-radius: 12px;
        border: 2px dashed #dc2626;
        padding: 1.5rem;
    }
    
    /* Result Card */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        border-top: 4px solid #dc2626;
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
        border-top: 3px solid #dc2626;
    }
    
    .feature-item:hover {
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-item h3,
    .feature-item p {
        color: #1a1a1a !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #dc2626, transparent);
        margin: 2rem 0;
    }
    
    /* Info Box */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #dc2626;
    }
    
    /* File Uploader - force background and label color with high specificity */
    div[data-testid="stFileUploader"] {
        background: #fff !important;
        border-radius: 14px !important;
        border: 2px dashed #dc2626 !important;
        padding: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] > div:first-child > label,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] .css-1c7y2kd {
        color: #b91c1c !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stFileUploader"] .uploadedFileName {
        color: #991b1b !important;
    }
        /* File Uploader - force background, label, info, and button color with high specificity */
        div[data-testid="stFileUploader"] {
            background: #fff !important;
            border-radius: 14px !important;
            border: 2px dashed #dc2626 !important;
            padding: 1.5rem !important;
            margin-bottom: 1.2rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] > div:first-child > label,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] .css-1c7y2kd {
            color: #b91c1c !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.01em !important;
        }
        div[data-testid="stFileUploader"] .uploadedFileName {
            color: #991b1b !important;
        }
        .uploader-info {
            color: #991b1b !important;
            font-size: 0.98rem !important;
            font-weight: 500 !important;
            margin-top: -0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stFileUploader"] button, div[data-testid="stFileUploader"] input[type="file"]::file-selector-button {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: 0 4px 6px -1px rgba(220,38,38,0.15) !important;
            transition: all 0.2s;
        }
        div[data-testid="stFileUploader"] button:hover, div[data-testid="stFileUploader"] input[type="file"]::file-selector-button:hover {
            background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%) !important;
            color: #fff !important;
        }
    
    /* Image Container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Styling */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #dc2626;
    }
    
    /* Step Instructions */
    .instruction-step {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-weight: 500;
        color: #1a1a1a;
    }
    
    /* Navbar Customization */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
</style>
"""

st.markdown(modern_css, unsafe_allow_html=True)


# --------- TERLUKA Branding Header ---------
logo_path = "logo-terluka.png"
col1, col2, col3 = st.columns([4, 2, 4])
with col1:
    st.write("")
with col2:
    st.image(logo_path, use_container_width=True)
with col3:
    st.write("")
st.markdown("<div style='margin-bottom:2.5rem;'></div>", unsafe_allow_html=True)

# --------- Load navbar ---------
pages = ["Beranda", "Deteksi"]

# Initialize state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Beranda"

navbar_choice = st_navbar(pages, selected=st.session_state.selected_page)

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


# ========== BERANDA PAGE ==========
if st.session_state['selected_page'] == "Beranda":
    # Hero Section with TERLUKA branding
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title" style="font-size:2.1rem;">Platform Deteksi Luka Bakar Kulit</div>
        <div class="hero-subtitle">
            <b>TERLUKA</b> adalah platform AI untuk mendeteksi dan mengklasifikasi tingkat luka bakar kulit secara akurat dan cepat.<br>
            Upload gambar Anda dan dapatkan diagnosis serta rekomendasi penanganan dalam hitungan detik.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Grid
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <h3>Akurat</h3>
            <p>Deteksi berbasis YOLOv5 dengan akurasi tinggi</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">⚡</div>
            <h3>Cepat</h3>
            <p>Hasil deteksi dalam hitungan detik</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">💊</div>
            <h3>Rekomendasi</h3>
            <p>Saran penanganan untuk setiap derajat</p>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📱</div>
            <h3>Mudah</h3>
            <p>Interface sederhana dan user-friendly</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Information Section
    st.markdown("## 📊 Klasifikasi Derajat Luka Bakar")
    st.markdown("Kenali berbagai tingkat keparahan luka bakar untuk penanganan yang tepat")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🟢 Luka Bakar Derajat 1 (Ringan)", expanded=False):
            st.markdown("""
            <div class="info-card">
                <h4>Karakteristik:</h4>
                <ul>
                    <li>Hanya mengenai lapisan epidermis (lapisan terluar kulit)</li>
                    <li>Kulit tampak kemerahan dan sedikit bengkak</li>
                    <li>Terasa nyeri dan sensitif saat disentuh</li>
                    <li>Tidak ada luka terbuka atau lepuhan</li>
                    <li>Biasanya sembuh dalam 3-6 hari</li>
                </ul>
                <p><strong>Contoh:</strong> Luka bakar matahari ringan, terkena setrika sebentar</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🟠 Luka Bakar Derajat 2 (Sedang)", expanded=False):
            st.markdown("""
            <div class="info-card">
                <h4>Karakteristik:</h4>
                <ul>
                    <li>Melibatkan lapisan epidermis dan dermis</li>
                    <li>Terbentuk lepuhan berisi cairan (blister)</li>
                    <li>Kulit tampak merah, basah, dan mengkilap</li>
                    <li>Nyeri yang lebih intens</li>
                    <li>Waktu penyembuhan 2-3 minggu</li>
                </ul>
                <p><strong>Contoh:</strong> Terkena air mendidih, api langsung dalam waktu singkat</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.expander("🔴 Luka Bakar Derajat 3 (Berat)", expanded=False):
            st.markdown("""
            <div class="info-card">
                <h4>Karakteristik:</h4>
                <ul>
                    <li>Merusak seluruh lapisan kulit hingga jaringan di bawahnya</li>
                    <li>Kulit tampak putih, hitam, atau hangus</li>
                    <li>Jaringan mati (nekrosis)</li>
                    <li>Mungkin tidak terasa nyeri karena kerusakan saraf</li>
                    <li>Memerlukan perawatan medis intensif dan cangkok kulit</li>
                </ul>
                <p><strong>Perhatian:</strong> Segera ke rumah sakit!</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("⚫ Luka Bakar Derajat 4 (Sangat Berat)", expanded=False):
            st.markdown("""
            <div class="info-card">
                <h4>Karakteristik:</h4>
                <ul>
                    <li>Menembus semua lapisan kulit hingga otot dan tulang</li>
                    <li>Kerusakan jaringan sangat ekstensif</li>
                    <li>Warna hitam atau hangus</li>
                    <li>Kondisi yang mengancam jiwa</li>
                    <li>Memerlukan perawatan intensif dan pembedahan</li>
                </ul>
                <p><strong>GAWAT DARURAT:</strong> Hubungi ambulans segera!</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div class="info-card" style="text-align: center; border-left: none; border-top: 4px solid #dc2626;">
        <h2>Siap Untuk Mendeteksi?</h2>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Navigasi ke halaman <strong>Deteksi</strong> untuk memulai analisis gambar luka bakar Anda
        </p>
    </div>
    """, unsafe_allow_html=True)


# ========== DETEKSI PAGE ==========
elif st.session_state['selected_page'] == "Deteksi":
    st.markdown("""
    <div class="hero-section" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">🔬 Deteksi Luka Bakar</div>
        <div class="hero-subtitle">Upload gambar luka bakar untuk analisis AI oleh <b>TERLUKA</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'detection_results' not in st.session_state:
        st.session_state['detection_results'] = []
    
    # Upload Section
    st.markdown("### 📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar luka bakar (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        key="uploaded_file",
        help="Upload gambar dengan kualitas baik untuk hasil deteksi optimal"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    # Custom info row for file size/type (fix indentation)
    st.markdown(
        '<div class="uploader-info">Limit 200MB per file • JPG, JPEG, PNG</div>', unsafe_allow_html=True
    )

    
    # Action Buttons
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        detect_btn = st.button("🚀 Jalankan Deteksi", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Reset Semua", use_container_width=True)
    
    if detect_btn:
        if uploaded_file is None:
            st.warning("⚠️ Harap upload gambar terlebih dahulu.")
        else:
            st.session_state['detection_results'] = []
            with st.spinner("🔍 Sedang menganalisis gambar..."):
                result = run_detection(uploaded_file)
                st.session_state['detection_results'].append(result)
            st.success("✅ Deteksi selesai!")
    
    if clear_btn:
        keys_to_clear = ['detection_results', 'uploaded_file']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Results Section
    if st.session_state['detection_results']:
        st.markdown("<hr>", unsafe_allow_html=True)
        
        for result in st.session_state['detection_results']:
            st.markdown(f"""
            <div class="result-card">
                <h2>📋 Hasil Analisis</h2>
                <p><strong>File:</strong> {result['filename']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if result['image_path'] and os.path.exists(result["image_path"]):
                col1, col2, col3 = st.columns([1.5, 2, 1.5])
                with col2:
                    st.image(result["image_path"], use_container_width=True, caption="Hasil Deteksi")
            
            if result['labels']:
                counts = Counter(result["labels"])
                
                st.markdown("### 🎯 Kelas yang Terdeteksi")
                badge_html = ""
                for label, count in counts.items():
                    badge_html += f'<span class="badge">{label} ({count}x)</span> '
                st.markdown(f'<div style="margin: 1rem 0;">{badge_html}</div>', unsafe_allow_html=True)
                
                # st.markdown("<hr>", unsafe_allow_html=True)
                
                st.markdown("## Rekomendasi Penanganan")
                
                for idx, label in enumerate(set(result["labels"]), 1):
                    st.markdown(f"### {idx}. {label}")
                    instructions = penanganan_dict.get(label, ["Tidak ada informasi penanganan."])
                    
                    for i, step in enumerate(instructions, 1):
                        st.markdown(f"""
                        <div class="instruction-step">
                            <strong>Langkah {i}:</strong> {step}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if idx < len(set(result["labels"])):
                        st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-card" style="margin-top: 2rem; background: #fef2f2;">
                    <strong>⚠️ Disclaimer:</strong> Rekomendasi ini bersifat informasi umum. 
                    Untuk luka bakar derajat 2 ke atas atau jika kondisi memburuk, segera konsultasikan dengan tenaga medis profesional.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Tidak ada luka bakar terdeteksi pada gambar.")
    else:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3>👋 Belum ada hasil deteksi</h3>
            <p>Upload gambar dan klik tombol <strong>Jalankan Deteksi</strong> untuk memulai analisis</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Halaman tidak ditemukan. Silakan kembali ke Beranda.")