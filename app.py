import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Hijaiyah Handwriting Detection",
    page_icon="✍️",
    layout="wide"
)

# --- INISIALISASI SESSION STATE (Untuk Riwayat) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- FUNGSI CACHING ---
# Cache model agar tidak di-load ulang setiap kali ada interaksi di UI
@st.cache_resource
def load_keras_model():
    return load_model("models/fixmodel.h5")

# Cache data JSON
@st.cache_data
def load_letter_data():
    try:
        with open('static/abjad_updated.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# --- FUNGSI PREPROCESSING ---
def preprocess_canvas_image(image_data):
    """Mengubah output canvas (RGBA array) menjadi format tensor Keras (1, 150, 150, 3)"""
    # Canvas mengembalikan numpy array (H, W, 4) uint8
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # Konversi ke RGB (menghilangkan alpha channel)
    img = img.convert('RGB')
    
    # Resize sesuai target model (150, 150)
    img = img.resize((150, 150))
    
    # Ubah ke array, normalisasi, dan expand dimensi
    img_array = np.array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = img_tensor / 255.0
    
    return img_tensor

# --- VARIABEL GLOBAL ---
CLASS_NAMES = ['ain', 'alif', 'ba', 'dal', 'dhod', 'dzal', 'dzho', 'fa', 'ghoin', 'ha', "ha'", 
               'hamzah', 'jim', 'kaf', 'kho', 'lam', 'lamalif', 'mim', 'nun', 'qof', 'ro', 
               'shod', 'sin', 'syin', 'ta', 'tho', 'tsa', 'wawu', 'ya', 'zain']

# Load Model & Data
try:
    model = load_keras_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

letter_data = load_letter_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ 画布设置 - Canvas Settings")
    stroke_width = st.slider("笔尖粗细 - Pen Thickness", 5, 25, 15)
    
    # Opsi warna penting karena beberapa model dilatih dengan latar hitam/tulisan putih, atau sebaliknya
    st.markdown("将颜色与模型的训练数据集相匹配。:")
    st.markdown("Match the colors to your model's training dataset:")
    stroke_color = st.color_picker("笔色 - Pen Color", "#000000") # Default Hitam
    bg_color = st.color_picker("背景颜色 - Background Color", "#FFFFFF")     # Default Putih
    
    st.markdown("---")
    st.info("该应用程序使用深度学习模型来检测手写的 Hijaiyah 字母。")
    st.info("This application uses a Deep Learning model to detect handwritten Hijaiyah letters.")

# --- TAMPILAN UTAMA ---
st.title("✍️ Hijaiyah 手写检测")
st.subheader("✍️ Hijaiyah Handwriting Detection")
st.markdown("在下面方框中画出 Hijaiyah 字母，然后点击 **预测** 查看结果。")
st.markdown("Draw the Hijaiyah letters in the box below, then click **Predict** to see the results.")

# Menggunakan container dan columns untuk layout
with st.container():
    col1, col2 = st.columns([2, 1]) # Kolom kiri lebih besar untuk canvas
    
    with col1:
        st.subheader("绘图区域 - Drawing Area")
        # Membuat canvas menggunakan streamlit-drawable-canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Warna isi jika menggunakan tool polygon
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=300,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        predict_btn = st.button("🔍 预测 Predict", type="primary", use_container_width=True)

    with col2:
        st.subheader("预测结果 - Prediction Results") 
        
        if predict_btn:
            if canvas_result.image_data is not None and model_loaded:
                # Cek apakah canvas kosong (hanya berisi satu warna)
                if len(np.unique(canvas_result.image_data)) <= 2: # Biasanya array kosong ada sedikit artifact
                    st.warning("请先在画布上画点什么！")
                    st.warning("Please draw something on the canvas first!")
                else:
                    with st.spinner("加工... Processing..."):
                        # Preprocessing
                        img_tensor = preprocess_canvas_image(canvas_result.image_data)
                        
                        # Prediksi
                        pred = model.predict(img_tensor)
                        class_idx = np.argmax(pred)
                        predicted_class = CLASS_NAMES[class_idx]
                        confidence = np.max(pred) * 100
                        
                        # Simpan ke riwayat
                        st.session_state['history'].append({
                            'huruf': predicted_class,
                            'akurasi': confidence
                        })
                        
                        # Tampilkan Hasil
                        st.success("分析成功！ Successfully analyzed!")
                        st.metric(label="检测到字母  Letter Detected", value=predicted_class.capitalize(), delta=f"Akurasi: {confidence:.2f}%", delta_color="normal")
                        
            elif not model_loaded:
                st.error("Failed to load model.")
        else:
            st.info("正在等待图像输入……")
            st.info("Waiting for image input...")

# --- RIWAYAT PREDIKSI ---
st.markdown("---")
with st.expander("🕰️ 本届会议预测历史 - This Session's Prediction History"):
    if len(st.session_state['history']) == 0:
        st.write("目前还没有任何预测。- There are no predictions yet.")
    else:
        # Menampilkan riwayat dari yang terbaru
        for idx, item in enumerate(reversed(st.session_state['history'])):
            st.markdown(f"**{len(st.session_state['history']) - idx}. {item['huruf'].capitalize()}** *(准确性 Accuracy: {item['akurasi']:.2f}%)*")

st.markdown("""
    <style>
    /* Menargetkan elemen gambar bawaan dari st.image */
    [data-testid="stImage"] img {
        height: 200px !important; 
        object-fit: contain !important; 
        background-color: white; 
        border-radius: 8px; /* Membuat sudut garis pinggir membulat */
        
        /* --- KODE BARU UNTUK GARIS PINGGIR --- */
        border: 2px solid #d3d3d3 !important; /* Ketebalan 2px, tipe solid, warna abu-abu muda */
        padding: 10px !important; /* Memberikan jarak antara gambar dan garis pinggir */
    }
    
    /* Menyelaraskan teks judul agar posisinya seragam */
    .hijaiyah-title {
        text-align: center;
        margin-bottom: 10px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("希贾亚字母列表")
st.subheader("List of Hijaiyah Letters\n\n")

# 1. Membaca data dari file JSON
# Pastikan file 'abjad_updated.json' berada di folder yang sama dengan file app.py ini
try:
    with open('static/abjad_updated.json', 'r', encoding='utf-8') as file:
        hijaiyah_data = json.load(file)
except FileNotFoundError:
    st.error("The abjad_updated.json file was not found. Make sure it is in the same directory.")
    hijaiyah_data = []

# 2. Membuat layout Grid
# Menggunakan 4 kolom agar tampilan rapi seperti galeri (bisa disesuaikan)
kolom_per_baris = 4
cols = st.columns(kolom_per_baris)

# 3. Menampilkan data (Looping)
for index, item in enumerate(hijaiyah_data):
    # Menentukan posisi kolom saat ini
    col = cols[index % kolom_per_baris]
    
    with col:
        # Menampilkan nama huruf sebagai header kecil
        st.markdown(f"<h3 style='text-align: center;'>{item['nama']}</h3>", unsafe_allow_html=True)
        
        # Menampilkan gambar
        # Fitur overlay sudah tertanam otomatis oleh Streamlit
        # Pengguna cukup mengklik ikon panah ganda pada gambar untuk melihatnya dalam mode layar penuh (overlay)
        try:
            st.image(item['link'], use_container_width=True)
        except Exception as e:
            st.warning(f"Image could not be loaded: {item['link']}")
            
        st.divider() # Garis pemisah opsional antar baris