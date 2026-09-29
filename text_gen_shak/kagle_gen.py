# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Device check for RTX 2050
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# 2. Load dataset locally (Make sure Shakespeare_data.csv is in the same folder)
csv_filename = 'Shakespeare_data.csv'
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found in current directory!")
    exit(1)

print("Loading dataset...")
data = pd.read_csv(csv_filename)
dataset = data['PlayerLine'].dropna().tolist()

# Convert lines to lowercase corpus (Local execution without MirroredStrategy)
corpus = [str(line).lower() for line in dataset]
print(f"Total lines loaded: {len(corpus):,}")

# 3. Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_to_token = tokenizer.word_index

def key_pair(num):    
    count = 0
    for key, value in word_to_token.items():
        if count >= num: 
            break
        print(f"'{key}': {value},")
        count += 1

print("\n--- Top Vocabulary Sample ---")
key_pair(10)

total_words = len(word_to_token) + 1
print(f"Total unique words: {total_words}")

# 4. Build input sequences (N-grams)
print("\nBuilding n-gram sequences...")
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print(f"Total sequences generated: {len(input_sequences):,}")

# Pad sequences
max_seq_len = max(len(x) for x in input_sequences)
print(f"Max sequence length: {max_seq_len}")

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

features = input_sequences[:, :-1]
labels = input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# 5. Take a safe slice for 16GB RAM / 4GB VRAM limits
n = 0.05  # Using 5% chunk of the massive dataset for fast local training
slice_size = int(len(features) * n)
print(f"Using a slice of {slice_size:,} samples for local training.")

features_chunk = features[:slice_size, :]
labels_chunk = labels[:slice_size, :]

# Save locally instead of /kaggle/working/
np.save('features.npy', features_chunk)
np.save('labels.npy', labels_chunk)

# Reload from local storage
features = np.load('features.npy')
labels = np.load('labels.npy')

# 6. Create the Bidirectional LSTM Model
def generator_model():
    tf.random.set_seed(42)
    model = Sequential([
        Embedding(total_words, 100, input_length=max_seq_len-1),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dense(total_words, activation='softmax')
    ])
    return model

model = generator_model()
model.compile(
    loss='categorical_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    metrics=['accuracy']
)
model.summary()

# 7. Train the Model
EPOCHS = 5  # Set to 5 for a quick test run; increase if desired
BATCH_SIZE = 64
print("\nStarting model training on GPU...")
history = model.fit(features, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model locally
model.save('test_generator.h5')
print("Model saved successfully as 'test_generator.h5'")

# 8. Text Generation Test Function
def test_generator(string, num):
    if len(string) == 0:
        print("Error: No word found")
        return
    for _ in range(num):
        token_list = tokenizer.texts_to_sequences([string])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding="pre")
        probabilities = model.predict(token_list, verbose=0)
        choice = np.random.choice([1, 2, 3])
        predicted = np.argsort(probabilities, axis=-1)[0][-choice]
        if predicted != 0:
            generated_word = tokenizer.index_word[predicted]
            string += " " + generated_word
    print("\n--- Generated Text Output ---")
    print(string)

test_generator("long live the king", 20)