import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ======================================================================================
# Konfigurasi Aplikasi dan Fungsi
# ======================================================================================

st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="😃", layout="wide")

# --- Kamus untuk Terjemahan dan Warna ---
kamus_emosi = {"sad": "Sedih", "disgust": "Jijik", "angry": "Marah", "neutral": "Netral", "fear": "Takut", "surprise": "Terkejut", "happy": "Senang"}

# BARU: Kamus warna untuk setiap emosi (format BGR untuk OpenCV)
warna_emosi = {
    "Marah": (0, 0, 255),       # Merah
    "Jijik": (0, 128, 0),       # Hijau Tua
    "Takut": (0, 165, 255),     # Oranye
    "Senang": (0, 255, 0),       # Hijau Terang
    "Netral": (255, 255, 255),    # Putih
    "Sedih": (255, 0, 0),       # Biru
    "Terkejut": (0, 255, 255)  # Kuning
}

# Muat model deteksi wajah
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error: Gagal memuat file 'haarcascade_frontalface_default.xml'.")

@st.cache_resource
def load_emotion_model():
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

def predict_emotion(image_pil, processor, model):
    inputs = processor(images=image_pil.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = model.config.id2label
    results_en = {labels[i]: prob.item() for i, prob in enumerate(probs[0])}
    results_id = {kamus_emosi.get(en_label, en_label): conf for en_label, conf in results_en.items()}
    return results_id

# --- Kelas untuk Memproses Frame Video (DIUBAH) ---
class EmotionPredictor(VideoTransformerBase):
    def __init__(self):
        self.processor, self.model = load_emotion_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            predictions = predict_emotion(pil_face, self.processor, self.model)
            
            if predictions:
                # Cari emosi teratas dan kepercayaannya
                top_emotion = max(predictions, key=predictions.get)
                confidence = predictions[top_emotion]

                # Dapatkan warna berdasarkan emosi dari kamus warna
                color = warna_emosi.get(top_emotion, (255, 255, 255)) # Default Putih

                # Buat label teks dengan persentase
                label_text = f"{top_emotion}: {confidence:.1%}"
                
                # Tentukan ukuran teks dan posisinya agar rapi
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_x = x
                text_y = y - 10
                
                # Gambar kotak latar belakang untuk teks agar mudah dibaca
                cv2.rectangle(img, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), color, -1)
                
                # Gambar kotak di sekeliling wajah dan teksnya
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2) # Teks warna hitam

        return img

# ======================================================================================
# Tampilan Antarmuka (UI) Streamlit
# ======================================================================================

st.title("😃 Aplikasi Deteksi Emosi Wajah")

with st.spinner('Sedang memuat model AI...'):
    processor, model = load_emotion_model()

tab1, tab2 = st.tabs(["**Deteksi dari Gambar**", "**Deteksi Real-Time dari Kamera**"])

with tab1:
    # ... (kode untuk tab 1 tetap sama) ...
    st.header("Unggah Gambar untuk Dianalisis")
    uploaded_file = st.file_uploader("Pilih sebuah gambar wajah...", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Gambar yang Diunggah', use_container_width=True)
        with col2:
            st.write("### **Hasil Analisis**")
            with st.spinner("Menganalisis emosi..."):
                predictions = predict_emotion(image, processor, model)
            emosi_teratas = max(predictions, key=predictions.get)
            kepercayaan_teratas = predictions[emosi_teratas]
            st.metric(label="**Emosi Terdeteksi**", value=emosi_teratas, delta=f"{kepercayaan_teratas:.2%}")
            st.write("#### Rincian Semua Probabilitas:")
            df_predictions = pd.DataFrame(list(predictions.items()), columns=['Emosi', 'Probabilitas']).sort_values(by='Probabilitas', ascending=False)
            st.bar_chart(df_predictions.set_index('Emosi'))

with tab2:
    st.header("Aktifkan Kamera untuk Deteksi Real-Time")
    st.write("Klik 'START' di bawah untuk memulai. Anda mungkin perlu memberikan izin akses kamera pada browser Anda.")
    webrtc_streamer(
        key="emotion_detection",
        video_processor_factory=EmotionPredictor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}, {"urls": ["stun:stun2.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.sidebar.info(
    "Aplikasi ini dibuat menggunakan Streamlit dan model dari Hugging Face. "
    "Performa real-time mungkin bervariasi tergantung koneksi dan kekuatan CPU."
)
