import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Start overall timer
total_start_time = time.time()

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}\n")

# --- TASK 1: DATA LOADING ---
t0 = time.time()
csv_filename = "Shakespeare_data.csv"
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found!")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(csv_filename)
corpus = [str(line).lower() for line in df['PlayerLine'].dropna().tolist()[:10000]] # 10k sample for speed
print(f"⏱️ Task 1 (Data Load): {time.time() - t0:.2f} seconds\n")

# --- TASK 2: TOKENIZATION & N-GRAM PREPARATION ---
t0 = time.time()
print("Tokenizing words and building sequences...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
print(f"⏱️ Task 2 (Tokenization & Padding): {time.time() - t0:.2f} seconds (Total sequences: {len(X):,})\n")

# --- TASK 3: MODEL BUILDING ---
t0 = time.time()
print("Building LSTM model architecture...")
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(f"⏱️ Task 3 (Model Compilation): {time.time() - t0:.2f} seconds\n")

# --- TASK 4: TRAINING (GPU / CPU EXECUTION) ---
t0 = time.time()
print("Training model...")
EPOCHS = 3
BATCH_SIZE = 64
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
training_time = time.time() - t0
print(f"⏱️ Task 4 (Model Training): {training_time:.2f} seconds ({training_time/EPOCHS:.2f} sec/epoch)\n")

print(f"🏁 Total Pipeline Time: {time.time() - total_start_time:.2f} seconds")