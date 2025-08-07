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
class_names = ["cataract", "diabetic", "glaucoma", "normal", "non-fundus"]
file_id = "1neG3g7T5xv-2BbB2OZ_LtiFuFmtBOzIa"

# --- Fungsi Unduh dari Google Drive Tanpa gdown ---
def download_from_gdrive(file_id, dest_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Unduh Model Jika Belum Ada ---
if not os.path.exists(model_weights_path):
    st.info("🔄 Mengunduh model dari Google Drive...")
    try:
        download_from_gdrive(file_id, model_weights_path)
        st.success("✅ Model berhasil diunduh.")
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

# --- Load Model ---
model = ViTForImageClassification.from_pretrained(model_ckpt)
model.classifier = torch.nn.Linear(model.classifier.in_features, len(class_names))
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device("cpu")))
model.eval()

processor = AutoImageProcessor.from_pretrained(model_ckpt)
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide")

if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Home"

halaman = st.sidebar.selectbox("Navigasi", ["Home", "Deteksi", "Tentang"], index=["Home", "Deteksi", "Tentang"].index(st.session_state["halaman"]))


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

        Bagian penting yang dapat diamati meliputi:
        - Diskus optik (saraf optik)
        - Makula
        - Pembuluh darah retina

        Pemeriksaan ini aman, cepat, dan sangat dianjurkan secara rutin terutama bagi penderita diabetes atau usia lanjut.
        """)

        if st.button("🚀 Mulai Deteksi Sekarang"):
            st.session_state["halaman"] = "Deteksi"
            st.rerun()

    with kolom_kanan:
        image_path = "mata.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, width=400)
        else:
            st.warning("Gambar tidak ditemukan. Pastikan file ada di direktori yang sama.")

    st.markdown("<h3 style='text-align: center;'>🖼️ Contoh Gambar Penyakit Mata</h3>", unsafe_allow_html=True)
    gambar_list = [("cataract.jpg", "Katarak"), ("diabetic.jpeg", "Retinopati Diabetik"), ("glaucoma.jpg", "Glaukoma"), ("normal.jpg", "Normal")]
    cols = st.columns(4)
    for col, (file, caption) in zip(cols, gambar_list):
        with col:
            st.image(file, caption=caption, use_container_width=True)
#halaman Deteksi
elif halaman == "Deteksi":
    st.markdown("<h1 style='text-align: center;'>👁️ Deteksi Penyakit Mata</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>📤 Unggah gambar fundus retina untuk mendapatkan hasil deteksi</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar fundus retina", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🔍 Mendeteksi penyakit..."):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)

            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

# Threshold kepercayaan minimum
            threshold = 0.8  

            if confidence < threshold:
                hasil_prediksi = f"Deteksi tidak meyakinkan ({confidence*100:.2f}%)"
                raw_label = None
            else:
                raw_label = class_names[predicted_class].capitalize()
                hasil_prediksi = f"{label_mapping.get(raw_label, raw_label)} ({confidence*100:.2f}%)"

        file_name = uploaded_file.name
        file_size_kb = uploaded_file.size / 1024
        image_width, image_height = image.size
        image_format = image.format if image.format else uploaded_file.type.split("/")[-1].upper()

        deskripsi = {
            "Normal": """
            Citra fundus menunjukkan struktur retina yang **normal**.  
            - **Cakram optik** terlihat jelas dengan batas tegas  
            - **Pembuluh darah** terdistribusi merata  
            - Tidak ada bercak merah (pendarahan) maupun putih (exudate)  
            """,

            "Cataract": """
            **Katarak** adalah kekeruhan pada lensa mata yang menyebabkan penglihatan menjadi buram.  
            Pada citra fundus:  
            - **Gambar tampak kabur** atau buram  
            - **Detail retina tidak terlihat jelas**  
            - Warna retina tampak kusam dan tidak tajam  
            """,

            "Glaucoma": """
            **Glaukoma** adalah kerusakan saraf optik akibat tekanan bola mata tinggi.  
            Pada citra fundus:  
            - **Cup-to-disc ratio membesar (>0.6)**  
            - **Tepi saraf optik menipis**  
            - Bisa terjadi **kehilangan pembuluh darah di area tepi cakram**  
            """,

            "Diabetic": """
            **Retinopati diabetik** adalah komplikasi retina akibat diabetes.  
            Pada citra fundus:  
            - Terlihat **microaneurysms** (bintik merah kecil)  
            - **Exudate putih** kekuningan  
            - **Pendarahan** (bercak merah gelap)  
            - Kadang terdapat **pembuluh darah abnormal** (neovaskularisasi)  
            """,

            "Non-fundus": """
            **Gambar yang Anda unggah bukan merupakan citra fundus retina manusia.**  
            Harap unggah gambar fundus yang valid untuk mendeteksi penyakit mata seperti katarak, glaukoma, atau retinopati diabetik.  
            Ciri citra fundus valid:  
            - Bentuk bulat seperti bola mata  
            - Terdapat cakram optik dan pembuluh darah retina  
            """
        }


        col1, col2 = st.columns([1.1, 1.7])
        with col1:
            st.image(image, caption="🖼️ Gambar Fundus", use_container_width=True)

        with col2:
            st.markdown(f"""
                 <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745;'>
                    <strong>Hasil Deteksi:</strong> <span style='color: #155724;'>{hasil_prediksi}</span>
                </div>
            """, unsafe_allow_html=True)


            st.markdown("### 📊 Probabilitas Klasifikasi")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(class_names, probabilities.numpy(), color="#1f77b4")
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probabilitas (%)")
            ax.invert_yaxis()
            st.pyplot(fig)

        col_info, col_desc = st.columns([1.1, 1.9])

        with col_info:
            with st.expander("ℹ️ Informasi Gambar", expanded=True):
                st.write(f"**Nama File:** {file_name}")
                st.write(f"**Ukuran File:** {file_size_kb:.2f} KB")
                st.write(f"**Resolusi:** {image_width} x {image_height} piksel")
                st.write(f"**Format:** {image_format}")

        with col_desc:
            if raw_label is not None:
                st.markdown("### 📝 Deskripsi")
                st.markdown(deskripsi.get(raw_label, "Deskripsi tidak tersedia."), unsafe_allow_html=True)
            else:
                st.markdown("### 📝 Deskripsi")
                st.warning("Tingkat kepercayaan terlalu rendah. Hasil deteksi tidak meyakinkan.")


        def buat_laporan_pdf():
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4

            margin = 50
            y = height - margin

            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, y, "Laporan Deteksi Penyakit Mata")

            c.setFont("Helvetica", 12)
            y -= 30
            c.drawString(margin, y, f"Nama File: {file_name}")
            y -= 20
            c.drawString(margin, y, f"Ukuran File: {file_size_kb:.2f} KB")
            y -= 20
            c.drawString(margin, y, f"Resolusi: {image_width} x {image_height}")
            y -= 20
            c.drawString(margin, y, f"Format: {image_format}")

            y -= 40
            c.drawString(margin, y, f"Hasil Deteksi: {hasil_prediksi}")
            y -= 20
            c.drawString(margin, y, f"Tingkat Kepercayaan: {confidence * 100:.2f}%")

            y -= 40
            c.drawString(margin, y, "Probabilitas Kelas:")
            for i, prob in enumerate(probabilities.numpy()):
                label = class_names[i].capitalize()
                label_indo = label_mapping.get(label, label)
                y -= 20
                c.drawString(margin + 20, y, f"- {label_indo}: {prob * 100:.2f}%")

            y -= 60
            styles = getSampleStyleSheet()
            style = styles["Normal"]
            deskripsi_text = f"<b>Deskripsi:</b><br/>{deskripsi.get(raw_label)}"
            p = Paragraph(deskripsi_text, style)
            frame = Frame(margin, y - 100, width - 2 * margin - 150, 100, showBoundary=0)
            frame.addFromList([p], c)

            temp_img_path = "temp_image.jpg"
            image.save(temp_img_path)
            c.drawImage(temp_img_path, width - 160, y + 50, width=110, height=110)

            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer

        pdf_buffer = buat_laporan_pdf()
        st.download_button("📄 Unduh Laporan Deteksi (PDF)", data=pdf_buffer, file_name="laporan_deteksi.pdf", mime="application/pdf")

    else:
        st.info("Silakan unggah gambar fundus retina terlebih dahulu.")

#Halaman Tentang
elif halaman == "Tentang":
    st.title("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari penelitian tugas akhir mengenai penerapan model Vision Transformer untuk klasifikasi penyakit mata dari citra fundus.

    **Di Kembangkan Oleh:**

    - **Nama**: Irvan Yudistiansyah  
    - **NIM**: 20210040082  
    - **Prodi**: Teknik Informatika  
    - **Universitas**: Nusa Putra Sukabumi  

    **Model:** google/vit-base-patch16-224  
    **Framework:** PyTorch & Hugging Face Transformers  
    **Antarmuka:** Streamlit  

    **Kelas Deteksi:**
    - **Katarak**: Kekeruhan lensa mata
    - **Glaukoma**: Kerusakan saraf optik karena tekanan bola mata
    - **Retinopati Diabetik**: Gangguan retina akibat komplikasi diabetes
    - **Normal**: Tidak terdeteksi gangguan
    """)


st.markdown("""
    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 13px; color: gray;">
    © 2025 | Dibuat oleh Irvan Yudistiansyah | Untuk keperluan edukasi & skripsi
    </div>
""", unsafe_allow_html=True)






