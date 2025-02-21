import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# โหลดชุดข้อมูล
file_path = "cleaned_hate_speech_dataset - cleaned_hate_speech_dataset.csv"
df = pd.read_csv(file_path)

# แสดงตัวอย่างข้อมูล
print(df.head())

# ตรวจสอบคอลัมน์
print(df.columns)

# คอลัมน์ข้อความและฉลาก
text_column = "tweet_en_clean"
label_column = "class"

# ลบค่าที่หายไป
df = df.dropna(subset=[text_column, label_column])

# แปลง labels เป็นค่าตัวเลข
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

# บันทึก LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# แบ่งชุดข้อมูล train และ test
X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[label_column], test_size=0.2, random_state=42)

# Tokenization
max_words = 10000
max_length = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# บันทึก Tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# แปลงข้อความเป็นลำดับตัวเลข
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# เติม padding
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding="post", truncating="post")
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding="post", truncating="post")

# สร้างโมเดล LSTM
model = keras.Sequential([
    keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# ฝึกโมเดล
epochs = 20
batch_size = 32

model.fit(X_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_padded, y_test))

# บันทึกโมเดล
model.save("hate_speech_modelv2.h5")
print("Model saved successfully.")

# ประเมินผล
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

