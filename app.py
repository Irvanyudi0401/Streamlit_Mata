import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

# ========== Konfigurasi ==========
model_ckpt = "google/vit-base-patch16-224"
model_weights_path = "vit_model.pth"
file_id = "1neG3g7T5xv-2BbB2OZ_LtiFuFmtBOzIa"
class_names = ["cataract", "diabetic", "glaucoma", "normal", "non-fundus"]

label_mapping = {
    "Cataract": "Katarak",
    "Diabetic": "Retinopati Diabetik",
    "Glaucoma": "Glaukoma",
    "Normal": "Mata Normal",
    "Non-fundus": "Bukan Gambar Fundus"
}

# ========== Unduh dari Google Drive ==========
def download_from_gdrive(file_id, dest_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ========== Unduh model jika belum tersedia ==========
if not os.path.exists(model_weights_path):
    st.info("🔄 Mengunduh model dari Google Drive...")
    try:
        download_from_gdrive(file_id, model_weights_path)
        size = os.path.getsize(model_weights_path) / (1024 * 1024)
        st.success(f"✅ Model berhasil diunduh ({size:.2f} MB)")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
        st.stop()

# ========== Load model ==========
try:
    # Inisialisasi model sesuai arsitektur pretrained
    model = ViTForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=len(class_names)
    )

    # Load bobot hasil training
    state_dict = torch.load(model_weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Processor untuk preprocessing gambar
    processor = AutoImageProcessor.from_pretrained(model_ckpt)

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ========== Streamlit Config ==========
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide")

# ========== Navigasi ==========
if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Home"

halaman = st.sidebar.selectbox(
    "Navigasi",
    ["Home", "Deteksi", "Tentang"],
    index=["Home", "Deteksi", "Tentang"].index(st.session_state["halaman"])
)

# ========== Halaman HOME ==========
if halaman == "Home":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata Menggunakan Citra Fundus</h1>", unsafe_allow_html=True)
    st.markdown("## 💬 Apa Itu Citra Fundus?")
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        Citra fundus retina adalah gambar bagian belakang mata (retina) yang diambil menggunakan kamera fundus.  
        Pemeriksaan ini penting dalam dunia medis karena dapat mendeteksi penyakit seperti:
        - **Katarak**
        - **Glaukoma**
        - **Retinopati Diabetik**
        """)

        if st.button("🚀 Mulai Deteksi Sekarang"):
            st.session_state["halaman"] = "Deteksi"
            st.rerun()

    with col2:
        if os.path.exists("mata.png"):
            st.image("mata.png", width=400)
        else:
            st.warning("🖼️ Gambar `mata.png` tidak ditemukan.")

    st.markdown("### 🖼️ Contoh Gambar Penyakit Mata")
    examples = [("cataract.jpg", "Katarak"), ("diabetic.jpeg", "Retinopati Diabetik"), ("glaucoma.jpg", "Glaukoma"), ("normal.jpg", "Normal")]
    cols = st.columns(4)
    for col, (img, label) in zip(cols, examples):
        if os.path.exists(img):
            col.image(img, caption=label, use_container_width=True)

# ========== Halaman DETEKSI ==========
elif halaman == "Deteksi":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>📤 Unggah gambar fundus retina untuk deteksi otomatis</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar fundus retina", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        file_name = uploaded_file.name
        file_size = uploaded_file.size / 1024
        width, height = image.size

        with st.spinner("🔍 Mendeteksi..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=0)

        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        threshold = 0.8
        if confidence < threshold:
            result = f"Deteksi tidak meyakinkan ({confidence*100:.2f}%)"
        else:
            label = class_names[pred_class].capitalize()
            result = f"{label_mapping.get(label, label)} ({confidence*100:.2f}%)"

        col1, col2 = st.columns([1.1, 1.7])
        with col1:
            st.image(image, caption="🖼️ Gambar Fundus", use_container_width=True)

        with col2:
            st.success(f"Hasil Deteksi: {result}")
            st.markdown("### 📊 Probabilitas Klasifikasi")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(class_names, probs.numpy(), color="#1f77b4")
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probabilitas")
            ax.invert_yaxis()
            st.pyplot(fig)

# ========== Halaman TENTANG ==========
elif halaman == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari penelitian tugas akhir mengenai penerapan model Vision Transformer untuk klasifikasi penyakit mata dari citra fundus.

    **Dikembangkan Oleh:**
    - **Nama**: Irvan Yudistiansyah  
    - **NIM**: 20210040082  
    - **Prodi**: Teknik Informatika  
    - **Universitas**: Nusa Putra Sukabumi

    **Teknologi yang digunakan:**
    - Model: `google/vit-base-patch16-224`
    - Framework: PyTorch, Hugging Face Transformers
    - Antarmuka: Streamlit

    **Kategori Deteksi:**
    - **Katarak**
    - **Glaukoma**
    - **Retinopati Diabetik**
    - **Normal**
    - **Bukan Gambar Fundus**
    """)

st.markdown("""
<hr style="margin-top: 50px;">
<div style="text-align: center; font-size: 13px; color: gray;">
&copy; 2025 | Dibuat oleh Irvan Yudistiansyah | Untuk keperluan edukasi & skripsi
</div>
""", unsafe_allow_html=True)

# ========== Jalankan Streamlit secara otomatis ==========
if __name__ == "__main__":
    import subprocess
    import sys
    # Pastikan script dijalankan via `streamlit run`
    if not any("streamlit" in arg for arg in sys.argv):
        subprocess.run(["streamlit", "run", __file__])
