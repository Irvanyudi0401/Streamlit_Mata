import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
import matplotlib.pyplot as plt
import numpy as np
import os
import gdown

# ========== KONFIGURASI ==========
MODEL_CKPT = "google/vit-base-patch16-224"
MODEL_WEIGHTS_PATH = "vit_model.pth"
GDRIVE_FILE_ID = "1hPaIgD9340jrUuE6iCrmE4NvSJk81sYQ"
CLASS_NAMES = ["cataract", "diabetic", "glaucoma", "normal", "non-fundus"]

LABEL_MAPPING = {
    "Cataract": "Katarak",
    "Diabetic": "Retinopati Diabetik",
    "Glaucoma": "Glaukoma",
    "Normal": "Mata Normal",
    "Non-fundus": "Bukan Gambar Fundus"
}

# ========== DOWNLOAD MODEL ==========
def download_model_from_gdrive(file_id, dest_path):
    """Unduh file dari Google Drive menggunakan gdown"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

if not os.path.exists(MODEL_WEIGHTS_PATH):
    st.info("🔄 Mengunduh model dari Google Drive...")
    try:
        download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_WEIGHTS_PATH)
        size = os.path.getsize(MODEL_WEIGHTS_PATH) / (1024 * 1024)
        st.success(f"✅ Model berhasil diunduh ({size:.2f} MB)")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
        st.stop()

# ========== LOAD MODEL ==========
try:
    model = ViTForImageClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(CLASS_NAMES)
    )

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")

    # Hapus layer terakhir jika jumlah output beda
    for key in ["classifier.weight", "classifier.bias"]:
        if key in state_dict and state_dict[key].shape[0] != len(CLASS_NAMES):
            del state_dict[key]

    # Load dengan strict=False agar layer yang dihapus tidak bikin error
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_CKPT)

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ========== KONFIGURASI STREAMLIT ==========
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide")

if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Home"

halaman = st.sidebar.selectbox(
    "Navigasi",
    ["Home", "Deteksi", "Tentang"],
    index=["Home", "Deteksi", "Tentang"].index(st.session_state["halaman"])
)

# ========== HALAMAN HOME ==========
if halaman == "Home":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata Menggunakan Citra Fundus</h1>", unsafe_allow_html=True)
    st.markdown("## 💬 Apa Itu Citra Fundus?")
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        Citra fundus retina adalah gambar bagian belakang mata (retina) yang diambil menggunakan kamera fundus.
        Pemeriksaan ini dapat mendeteksi:
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
    examples = [
        ("cataract.jpg", "Katarak"),
        ("diabetic.jpeg", "Retinopati Diabetik"),
        ("glaucoma.jpg", "Glaukoma"),
        ("normal.jpg", "Normal")
    ]
    cols = st.columns(4)
    for col, (img, label) in zip(cols, examples):
        if os.path.exists(img):
            col.image(img, caption=label, use_container_width=True)

# ========== HALAMAN DETEKSI ==========
elif halaman == "Deteksi":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>📤 Unggah gambar fundus retina untuk deteksi otomatis</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar fundus retina", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

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
            label = CLASS_NAMES[pred_class].capitalize()
            result = f"{LABEL_MAPPING.get(label, label)} ({confidence*100:.2f}%)"

        col1, col2 = st.columns([1.1, 1.7])
        with col1:
            st.image(image, caption="🖼️ Gambar Fundus", use_container_width=True)

        with col2:
            st.success(f"Hasil Deteksi: {result}")
            st.markdown("### 📊 Probabilitas Klasifikasi")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(CLASS_NAMES, probs.numpy(), color="#1f77b4")
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probabilitas")
            ax.invert_yaxis()
            st.pyplot(fig)

# ========== HALAMAN TENTANG ==========
elif halaman == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari penelitian tugas akhir mengenai penerapan Vision Transformer untuk klasifikasi penyakit mata dari citra fundus.

    **Dikembangkan Oleh:**
    - **Nama**: Irvan Yudistiansyah
    - **NIM**: 20210040082
    - **Prodi**: Teknik Informatika
    - **Universitas**: Nusa Putra Sukabumi

    **Teknologi yang digunakan:**
    - Model: `google/vit-base-patch16-224`
    - Framework: PyTorch, Hugging Face Transformers
    - Antarmuka: Streamlit
    """)

st.markdown("""
<hr style="margin-top: 50px;">
<div style="text-align: center; font-size: 13px; color: gray;">
&copy; 2025 | Dibuat oleh Irvan Yudistiansyah | Untuk keperluan edukasi & skripsi
</div>
""", unsafe_allow_html=True)




