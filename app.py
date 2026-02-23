import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Pengenalan Huruf Hijaiyah",
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
    st.error(f"Gagal memuat model: {e}")

letter_data = load_letter_data()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan Canvas")
    stroke_width = st.slider("Ketebalan Pena", 5, 25, 15)
    
    # Opsi warna penting karena beberapa model dilatih dengan latar hitam/tulisan putih, atau sebaliknya
    st.markdown("Sesuaikan warna dengan dataset latih model Anda:")
    stroke_color = st.color_picker("Warna Pena", "#000000") # Default Hitam
    bg_color = st.color_picker("Warna Latar", "#FFFFFF")     # Default Putih
    
    st.markdown("---")
    st.info("Aplikasi ini menggunakan model Deep Learning untuk mendeteksi tulisan tangan huruf Hijaiyah.")

# --- TAMPILAN UTAMA ---
st.title("✍️ Deteksi Tulisan Tangan Hijaiyah")
st.markdown("Gambarlah huruf Hijaiyah di dalam kotak di bawah ini, lalu klik **Prediksi** untuk melihat hasilnya.")

# Menggunakan container dan columns untuk layout
with st.container():
    col1, col2 = st.columns([2, 1]) # Kolom kiri lebih besar untuk canvas
    
    with col1:
        st.subheader("Area Gambar")
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
        
        predict_btn = st.button("🔍 Prediksi Huruf", type="primary", use_container_width=True)

    with col2:
        st.subheader("Hasil Prediksi")
        
        if predict_btn:
            if canvas_result.image_data is not None and model_loaded:
                # Cek apakah canvas kosong (hanya berisi satu warna)
                if len(np.unique(canvas_result.image_data)) <= 2: # Biasanya array kosong ada sedikit artifact
                    st.warning("Silakan gambar sesuatu di canvas terlebih dahulu!")
                else:
                    with st.spinner("Memproses..."):
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
                        st.success("Berhasil dianalisis!")
                        st.metric(label="Huruf Terdeteksi", value=predicted_class.capitalize(), delta=f"Akurasi: {confidence:.2f}%", delta_color="normal")
                        
                        # Tampilkan penjelasan singkat dari JSON (jika ada)
                        if letter_data and predicted_class in letter_data:
                            with st.expander(f"📖 Tentang huruf {predicted_class.capitalize()}", expanded=True):
                                st.write(letter_data[predicted_class])
            elif not model_loaded:
                st.error("Model belum siap/gagal dimuat.")
        else:
            st.info("Menunggu input gambar...")

# --- RIWAYAT PREDIKSI ---
st.markdown("---")
with st.expander("🕰️ Riwayat Prediksi Sesi Ini"):
    if len(st.session_state['history']) == 0:
        st.write("Belum ada prediksi.")
    else:
        # Menampilkan riwayat dari yang terbaru
        for idx, item in enumerate(reversed(st.session_state['history'])):
            st.markdown(f"**{len(st.session_state['history']) - idx}. {item['huruf'].capitalize()}** *(Akurasi: {item['akurasi']:.2f}%)*")