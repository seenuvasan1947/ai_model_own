import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Start overall timer
total_start_time = time.time()

# 1. Device Check (RTX 2050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Using Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}\n")

# --- TASK 1: DATA LOADING ---
t0 = time.time()
csv_filename = "Shakespeare_data.csv"
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found!")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(csv_filename)
corpus = [str(line).lower() for line in df['PlayerLine'].dropna().tolist()[:10000]] # 10k sample matching your test
print(f"⏱️ Task 1 (Data Load): {time.time() - t0:.2f} seconds\n")

# --- TASK 2: TOKENIZATION & N-GRAM PREPARATION ---
t0 = time.time()
print("Tokenizing words and building sequences...")

# Build vocabulary manually for lightweight PyTorch tokenization
words = " ".join(corpus).split()
vocab = sorted(list(set(words)))
word_to_int = {w: i + 1 for i, w in enumerate(vocab)} # Reserve 0 for padding
word_to_int['<PAD>'] = 0
int_to_word = {i: w for w, i in word_to_int.items()}
total_words = len(word_to_int)

input_sequences = []
for line in corpus:
    token_list = [word_to_int.get(w, 0) for w in line.split()]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 5

# Pad sequences manually ('pre' padding to match tf.keras behavior)
padded_sequences = []
for seq in input_sequences:
    padded = [0] * (max_sequence_len - len(seq)) + seq
    padded_sequences.append(padded)

padded_array = np.array(padded_sequences)
X = padded_array[:, :-1]
y = padded_array[:, -1] # PyTorch CrossEntropyLoss expects class indices, not one-hot vectors

print(f"⏱️ Task 2 (Tokenization & Padding): {time.time() - t0:.2f} seconds (Total sequences: {len(X):,})\n")

# --- TASK 3: MODEL BUILDING ---
t0 = time.time()
print("Building PyTorch LSTM model architecture...")

class EquivalentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim_1, hidden_dim_2):
        super(EquivalentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim_1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.fc = nn.Linear(hidden_dim_2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        # Match TensorFlow's dense projection on the final sequence state
        x = self.fc(x[:, -1, :])
        return x

model = EquivalentLSTM(vocab_size=total_words, embedding_dim=100, hidden_dim_1=150, hidden_dim_2=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

print(f"⏱️ Task 3 (Model Compilation): {time.time() - t0:.2f} seconds\n")

# --- TASK 4: TRAINING (GPU ACCELERATED) ---
t0 = time.time()
print("Training model on GPU...")
EPOCHS = 3
BATCH_SIZE = 64
dataset_size = len(X)

# Convert numpy arrays to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

model.train()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    permutation = torch.randperm(dataset_size)
    
    for i in range(0, dataset_size, BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_tensor[indices].to(device), y_tensor[indices].to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / (dataset_size / BATCH_SIZE)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f} seconds")

training_time = time.time() - t0
print(f"\n⏱️ Task 4 (Model Training): {training_time:.2f} seconds ({training_time/EPOCHS:.2f} sec/epoch)")
print(f"🏁 Total Pipeline Time: {time.time() - total_start_time:.2f} seconds")