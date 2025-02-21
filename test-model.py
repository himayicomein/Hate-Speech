import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# โหลดโมเดลและออบเจ็กต์ที่บันทึกไว้
loaded_model = keras.models.load_model("hate_speech_modelv2.h5")

# Corrected: Swap loaded_tokenizer and loaded_label_encoder
with open("label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)  # This should be loaded_label_encoder

with open("tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)  # This should be loaded_tokenizer

# Mapping คลาสเป็นชื่อที่อ่านง่าย
class_mapping = {
    0: "0 - Hate Speech",
    1: "1 - Offensive Language",
    2: "2 - Neither"
}

# ฟังก์ชันทำนายข้อความใหม่
def predict_text(text):
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")  # max_length = 100 ตามที่ใช้ตอนเทรน
    prediction = loaded_model.predict(padded)
    predicted_class = np.argmax(prediction)

    # ตรวจสอบว่าคลาสที่ทำนายอยู่ใน class_mapping หรือไม่
    return class_mapping.get(predicted_class, "Unknown")

# ทดสอบการทำนาย
sample_text = " As a woman you shouldnt complain about cleaning up your house as a man you should always take the trash out "
print(f"Prediction: {predict_text(sample_text)}")
