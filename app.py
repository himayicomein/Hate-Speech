import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import time
from googletrans import Translator

# โมเดล
loaded_model = keras.models.load_model("C:/Users/Narissara/OneDrive/Desktop/NLP Project/hate_speech_modelv2.h5")

with open("C:/Users/Narissara/OneDrive/Desktop/NLP Project/label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)

with open("C:/Users/Narissara/OneDrive/Desktop/NLP Project/tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)

#ชื่อ ฉสฟหห
class_mapping = {
    0: "🚨 Hate Speech",
    1: "⚠️ Offensive Language",
    2: "✅ Neither"
}

# คำแนะนำสำหรับข้อความที่ไม่ดี
advice_mapping = {
    "🚨 Hate Speech": "โปรดหลีกเลี่ยงการใช้ถ้อยคำที่แสดงความเกลียดชัง และพยายามใช้ภาษาที่สร้างสรรค์",
    "⚠️ Offensive Language": "ข้อความของคุณอาจมีเนื้อหาที่ไม่เหมาะสม โปรดพิจารณาปรับเปลี่ยนคำพูดให้สุภาพขึ้น"
}

# โหลดประวัติอินพุต
try:
    with open("history.json", "r", encoding="utf-8") as f:
        input_history = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    input_history = []

# ฟังก์ชันทำนายข้อความใหม่ และบันทึกอินพุต
def predict_text(text):
    # แปลข้อความเป็นภาษาอังกฤษแบบตรงตัว
    translator = Translator()
    translated_text = translator.translate(text, src="auto", dest="en").text
    st.write(f"📝 Translated Text: {translated_text}")
    
    # แปลงเป็นลำดับและทำ Padding
    sequence = loaded_tokenizer.texts_to_sequences([translated_text])
    padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
    prediction = loaded_model.predict(padded)
    predicted_class = np.argmax(prediction)
    result = class_mapping.get(predicted_class, "Unknown")
    
    # บันทึกข้อมูลอินพุต
    input_data = {"original_text": text, "translated_text": translated_text, "prediction": result}
    input_history.append(input_data)
    with open("history.json", "w", encoding="utf-8") as f:
        json.dump(input_history, f, indent=4, ensure_ascii=False)
    
    return result

# ตกแต่ง UI Streamlit
st.set_page_config(page_title="Hate Speech Detection", page_icon="🚨", layout="centered")

st.markdown("""  
    <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
            font-family: Arial, sans-serif;
        }
        .stTextArea textarea {
            background-color: #333;
            color: #FFF;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }
    </style>
    <h1 style="text-align:center; color:#FF4B4B;">⚠️ Hate Speech Detector</h1>
    <p style="text-align:center; font-size:18px; color:#FFFFFF;">ใส่ข้อความที่ต้องการตรวจสอบ และดูผลลัพธ์การจำแนกประเภท</p>
    <hr>
    """, unsafe_allow_html=True)

# ให้ผู้ใช้กรอกข้อความ
st.markdown("<h3>📝 Input your text below:</h3>", unsafe_allow_html=True)
user_input = st.text_area("💬 Enter your text:", height=150)

# ถ้ามีการกรอกข้อความแล้วให้ทำนาย
if st.button("🔍 Analyze Text"):
    if user_input.strip():
        prediction = predict_text(user_input)
        
        # สร้างแถบความคืบหน้า (Progress Bar) ก่อนทำการทำนาย
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        if "Hate Speech" in prediction:
            st.error(f"{prediction}")
            st.warning(f"💡 Suggestion: {advice_mapping[prediction]}")
        elif "Offensive Language" in prediction:
            st.warning(f"{prediction}")
            st.info(f"💡 Suggestion: {advice_mapping[prediction]}")
        else:
            st.success(f"{prediction}")
            st.balloons()
    else:
        st.warning("⚠️ กรุณาใส่ข้อความก่อนกดปุ่มวิเคราะห์!")

# แสดงประวัติอินพุต
st.markdown("<h3>📜 History</h3>", unsafe_allow_html=True)
if input_history:
    for idx, entry in enumerate(reversed(input_history[-10:]), 1):  # แสดงรายการล่าสุด 10 รายการ
        with st.expander(f"🔹 Entry {idx}"):
            st.write(f"**Original Text:** {entry['original_text']}")
            st.write(f"**Translated Text:** {entry['translated_text']}")
            st.write(f"**Prediction:** {entry['prediction']}")
else:
    st.info("ยังไม่มีประวัติการทดสอบข้อความ")
