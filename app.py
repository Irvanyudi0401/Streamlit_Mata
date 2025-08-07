import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet

# --- Konfigurasi Model ---
model_ckpt = "google/vit-base-patch16-224"
model_weights_path = "vit_model.pth"
file_id = "1neG3g7T5xv-2BbB2OZ_LtiFuFmtBOzIa"
class_names = ["cataract", "diabetic", "glaucoma", "normal", "non-fundus"]

# --- Fungsi Aman Unduh Google Drive ---
def download_from_gdrive(file_id, dest_path):
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(URL, stream=True)

    head = response.content[:512].lower()
    if b"<html" in head or b"google" in head:
        raise Exception("⚠️ Gagal mengunduh: bukan file model valid dari Google Drive.")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Unduh model jika belum tersedia ---
if not os.path.exists(model_weights_path):
    st.info("🔄 Mengunduh model dari Google Drive...")
    try:
        download_from_gdrive(file_id, model_weights_path)
        size_mb = os.path.getsize(model_weights_path) / (1024 * 1024)
        st.success(f"✅ Model berhasil diunduh ({size_mb:.2f} MB)")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
        st.stop()

# --- Mapping label ke Bahasa Indonesia ---
label_mapping = {
    "Cataract": "Katarak",
    "Diabetic": "Retinopati Diabetik",
    "Glaucoma": "Glaukoma",
    "Normal": "Mata Normal",
    "Non-fundus": "Bukan Gambar Fundus"
}

# --- Load model Vision Transformer ---
try:
    model = ViTForImageClassification.from_pretrained(model_ckpt)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names))
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

processor = AutoImageProcessor.from_pretrained(model_ckpt)
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide")

# --- Navigasi ---
if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Home"

halaman = st.sidebar.selectbox("Navigasi", ["Home", "Deteksi", "Tentang"], index=["Home", "Deteksi", "Tentang"].index(st.session_state["halaman"]))

# ================== HALAMAN HOME ==================
if halaman == "Home":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata Menggunakan Citra Fundus</h1>", unsafe_allow_html=True)
    st.markdown("## 💬 Apa Itu Citra Fundus?")

    kolom_kiri, kolom_kanan = st.columns(2)

    with kolom_kiri:
        st.write("""
        Citra fundus retina adalah gambar bagian belakang mata (retina) yang diambil menggunakan kamera fundus. 
        Pemeriksaan ini penting dalam dunia medis karena dapat mendeteksi berbagai penyakit mata secara dini, seperti:
        - **Glaukoma**
        - **Retinopati Diabetik**
        - **Katarak**
        """)

        if st.button("🚀 Mulai Deteksi Sekarang"):
            st.session_state["halaman"] = "Deteksi"
            st.rerun()

    with kolom_kanan:
        if os.path.exists("mata.png"):
            st.image("mata.png", width=400)
        else:
            st.warning("🖼️ File `mata.png` tidak ditemukan.")

    st.markdown("### 🖼️ Contoh Gambar Fundus")
    contoh = [("cataract.jpg", "Katarak"), ("diabetic.jpeg", "Retinopati Diabetik"), ("glaucoma.jpg", "Glaukoma"), ("normal.jpg", "Normal")]
    cols = st.columns(4)
    for col, (f, label) in zip(cols, contoh):
        if os.path.exists(f):
            col.image(f, caption=label, use_container_width=True)

# ================== HALAMAN DETEKSI ==================
elif halaman == "Deteksi":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>📤 Unggah gambar fundus retina untuk deteksi otomatis</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar fundus retina", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        file_name = uploaded_file.name
        file_size_kb = uploaded_file.size / 1024
        image_width, image_height = image.size
        image_format = image.format

        with st.spinner("🔍 Mendeteksi..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)

        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        threshold = 0.8
        if confidence < threshold:
            hasil_prediksi = f"Deteksi tidak meyakinkan ({confidence*100:.2f}%)"
            raw_label = None
        else:
            raw_label = class_names[predicted_class].capitalize()
            hasil_prediksi = f"{label_mapping.get(raw_label, raw_label)} ({confidence*100:.2f}%)"

        col1, col2 = st.columns([1.1, 1.7])
        with col1:
            st.image(image, caption="🖼️ Gambar Fundus", use_container_width=True)

        with col2:
            st.success(f"Hasil Deteksi: {hasil_prediksi}")
            st.markdown("### 📊 Probabilitas")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(class_names, probabilities.numpy(), color="#1f77b4")
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probabilitas")
            ax.invert_yaxis()
            st.pyplot(fig)

        if raw_label:
            st.markdown("### 📝 Deskripsi")
            st.write(f"Ciri-ciri dari kelas **{label_mapping[raw_label]}** akan ditampilkan di sini.")

# ================== HALAMAN TENTANG ==================
elif halaman == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari penelitian tugas akhir mengenai penerapan model Vision Transformer untuk klasifikasi penyakit mata dari citra fundus.

    **Dikembangkan Oleh:**
    - **Nama**: Irvan Yudistiansyah  
    - **NIM**: 20210040082  
    - **Prodi**: Teknik Informatika  
    - **Universitas**: Nusa Putra Sukabumi

    **Teknologi:**
    - Model: `google/vit-base-patch16-224`
    - Framework: PyTorch, Hugging Face Transformers
    - UI: Streamlit

    **Kelas Deteksi:**
    - **Katarak**: Kekeruhan lensa mata
    - **Glaukoma**: Kerusakan saraf optik karena tekanan bola mata
    - **Retinopati Diabetik**: Gangguan retina akibat komplikasi diabetes
    - **Normal**: Tidak terdeteksi gangguan

    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 13px; color: gray;">
    © 2025 | Dibuat oleh Irvan Yudistiansyah | Untuk keperluan edukasi & skripsi
    </div>
    """, unsafe_allow_html=True)


