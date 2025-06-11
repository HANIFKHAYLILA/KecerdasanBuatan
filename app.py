import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from collections import Counter

# ======================================================================================
# Konfigurasi Aplikasi dan Fungsi
# ======================================================================================

st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="😃", layout="wide")

# --- Kamus untuk Terjemahan dan Warna ---
kamus_emosi = {"sad": "Sedih", "disgust": "Jijik", "angry": "Marah", "neutral": "Netral", "fear": "Takut", "surprise": "Terkejut", "happy": "Senang"}

warna_emosi = {
    "Marah": (0, 0, 255),       # Merah
    "Jijik": (0, 128, 0),       # Hijau Tua
    "Takut": (0, 165, 255),     # Oranye
    "Senang": (0, 255, 0),       # Hijau Terang
    "Netral": (255, 255, 255),    # Putih
    "Sedih": (255, 0, 0),       # Biru
    "Terkejut": (0, 255, 255)  # Kuning
}

# Muat model deteksi wajah Haar Cascade
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error: Gagal memuat file 'haarcascade_frontalface_default.xml'.")

# Fungsi untuk memuat model emosi (dengan caching) berdasarkan nama
@st.cache_resource
def load_model(model_name):
    """Memuat model dan processor dari Hugging Face berdasarkan nama."""
    st.info(f"Memuat model: {model_name}...")
    # 'use_fast=False' untuk menghilangkan peringatan
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    st.info("Model selesai dimuat.")
    return processor, model

# Fungsi prediksi yang membaca label dinamis dari model
def predict_emotion(image_pil, processor, model):
    labels = model.config.id2label
    inputs = processor(images=image_pil.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    results_en = {labels[i]: prob.item() for i, prob in enumerate(probs[0])}
    results_id = {kamus_emosi.get(en_label, en_label): conf for en_label, conf in results_en.items()}
    return results_id

# Kelas untuk memproses frame video, menerima model sebagai argumen
class EmotionPredictor(VideoTransformerBase):
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            predictions = predict_emotion(pil_face, self.processor, self.model)
            if predictions:
                top_emotion = max(predictions, key=predictions.get)
                confidence = predictions[top_emotion]
                color = warna_emosi.get(top_emotion, (255, 255, 255))
                label_text = f"{top_emotion}: {confidence:.1%}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x, y - text_height - baseline - 5), (x + text_width, y), color, -1)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return img

# ======================================================================================
# Tampilan Antarmuka (UI) Streamlit
# ======================================================================================

st.title("😃 Aplikasi Deteksi Emosi Wajah")

# --- Bagian Sidebar untuk Pilihan Model dan Kontrol Input ---
st.sidebar.title("⚙️ Pengaturan")

# Pilihan Model
model_options = {
    "Dima's Emotions (Default)": "dima806/facial_emotions_image_detection",
    "Ricky's Emotions": "RickyIG/emotion_face_image_classification"
}
selected_model_key = st.sidebar.radio(
    "Pilih model klasifikasi emosi:",
    list(model_options.keys())
)
selected_model_name = model_options[selected_model_key]

# Pemuatan model berdasarkan pilihan di sidebar
with st.spinner(f"Memuat model '{selected_model_key}'..."):
    processor, model = load_model(selected_model_name)

st.sidebar.markdown("---")
st.sidebar.title("🖼️ Kontrol Input Gambar")
st.sidebar.info("Gunakan panel ini untuk mengunggah gambar yang ingin dianalisis.")
uploaded_file = st.sidebar.file_uploader("Pilih sebuah gambar wajah...", type=["jpg", "jpeg", "png"], key="uploader")
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat menggunakan Streamlit dan model dari Hugging Face.")

# --- Bagian Utama dengan Tabs ---
tab1, tab2 = st.tabs(["**Deteksi dari Gambar**", "**Deteksi Real-Time dari Kamera**"])

with tab1:
    st.header("Analisis Gambar dengan Multi-Wajah")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        all_face_predictions = []

        for (x, y, w, h) in faces:
            face_roi = img_cv[y:y+h, x:x+w]
            pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            predictions = predict_emotion(pil_face, processor, model)
            all_face_predictions.append(predictions)

            if predictions:
                top_emotion = max(predictions, key=predictions.get)
                confidence = predictions[top_emotion]
                color = warna_emosi.get(top_emotion, (255, 255, 255))
                label_text = f"{top_emotion}: {confidence:.1%}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_cv, (x, y - text_height - baseline), (x + text_width, y), color, -1)
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img_cv, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption='Gambar dengan Deteksi Emosi', use_container_width=True)

        st.header("Ringkasan Analisis")
        if len(faces) > 0:
            st.write(f"**Total Wajah Terdeteksi:** **{len(faces)}**")
            emotion_counts = Counter([max(p, key=p.get) for p in all_face_predictions if p])
            df_summary = pd.DataFrame(emotion_counts.items(), columns=['Emosi', 'Jumlah Terdeteksi']).sort_values(by='Jumlah Terdeteksi', ascending=False)
            st.dataframe(df_summary, use_container_width=True)

            st.markdown("---")
            st.subheader("Lihat Rincian Probabilitas per Wajah")
            face_options = [f"Wajah {i+1}" for i in range(len(faces))]
            selected_face_option = st.selectbox("Pilih wajah untuk melihat detail:", face_options)
            
            selected_index = face_options.index(selected_face_option)
            selected_predictions = all_face_predictions[selected_index]
            
            df_detail = pd.DataFrame(list(selected_predictions.items()), columns=['Emosi', 'Probabilitas']).sort_values(by='Probabilitas', ascending=False)
            st.write(f"**Rincian untuk {selected_face_option}:**")
            st.bar_chart(df_detail.set_index('Emosi'))
        else:
            st.warning("Tidak ada wajah yang terdeteksi pada gambar ini.")
    else:
        st.info("Silakan unggah sebuah gambar melalui panel di sebelah kiri.")

with tab2:
    st.header("Aktifkan Kamera untuk Deteksi Real-Time")
    st.write("Klik 'START' di bawah untuk memulai.")
    webrtc_streamer(
        key="emotion_detection",
        video_processor_factory=lambda: EmotionPredictor(processor=processor, model=model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}, {"urls": ["stun:stun2.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )