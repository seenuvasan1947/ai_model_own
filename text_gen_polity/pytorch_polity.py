import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Start overall timer
total_start_time = time.time()

# 1. Device Check (Enables CUDA GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- TASK 1: DATA LOADING FROM TXT FILE ---
t0 = time.time()
txt_filename = "indian_polity.txt"  # <-- Make sure your text file name matches this

if not os.path.exists(txt_filename):
    print(f"Error: '{txt_filename}' not found!")
    print("Please place your text file in the working directory.")
    exit(1)

print(f"Loading dataset from {txt_filename}...")
with open(txt_filename, "r", encoding="utf-8") as f:
    text_data = f.read()

# Build corpus from text lines
corpus = [line.strip().lower() for line in text_data.split("\n") if line.strip()]
print(f"Loaded {len(corpus):,} lines of text.")
print(f"⏱️ Task 1 (Data Load): {time.time() - t0:.2f} seconds\n")

# --- TASK 2: TOKENIZATION & N-GRAM PREPARATION ---
t0 = time.time()
print("Tokenizing words and building sequences...")

words = " ".join(corpus).split()
if not words:
    print("Error: The text file contains no readable words.")
    exit(1)

vocab = sorted(list(set(words)))
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for i, w in enumerate(vocab)}
total_words = len(vocab)
print(f"Total unique words: {total_words}")

input_sequences = []
for line in corpus:
    token_list = [word_to_int[w] for w in line.split() if w in word_to_int]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 5
padded_sequences = []
for seq in input_sequences:
    padded = [0] * (max_sequence_len - len(seq)) + seq
    padded_sequences.append(padded)

padded_sequences = np.array(padded_sequences)
X = torch.tensor(padded_sequences[:, :-1], dtype=torch.long)
y = torch.tensor(padded_sequences[:, -1], dtype=torch.long)

print(f"⏱️ Task 2 (Tokenization & Padding): {time.time() - t0:.2f} seconds (Total sequences: {len(X):,})\n")

# --- TASK 3: PYTORCH LSTM MODEL ARCHITECTURE ---
class WordLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordLSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

EMBEDDING_DIM = 100
HIDDEN_DIM = 150

model = WordLSTMGenerator(total_words, EMBEDDING_DIM, HIDDEN_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
print(f"⏱️ Task 3 (Model Initialized on {device.type.upper()})\n")

# --- TASK 4: TRAINING LOOP WITH LIVE BATCH PROGRESS ---
t0 = time.time()
EPOCHS = 10
BATCH_SIZE = 64
dataset_size = len(X)
total_batches = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Training PyTorch LSTM model on {device.type.upper()}...")
model.train()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0
    permutation = torch.randperm(dataset_size)
    
    for batch_idx, i in enumerate(range(0, dataset_size, BATCH_SIZE)):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X[indices].to(device), y[indices].to(device)
        
        optimizer.zero_grad()
        output, _ = model(batch_x, None)
        loss = criterion(output, batch_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Live batch progress update on the same line
        progress = (batch_idx + 1) / total_batches * 100
        print(f"\rEpoch [{epoch+1}/{EPOCHS}] Progress: {progress:5.1f}% | Batch {batch_idx+1}/{total_batches}", end="", flush=True)

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_batches
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Completed | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

training_time = time.time() - t0
print(f"⏱️ Task 4 (Total Training Time): {training_time:.2f} seconds")
print(f"🏁 Total Pipeline Time: {time.time() - total_start_time:.2f} seconds")

# --- SAVE MODEL AND DICTIONARIES ---
MODEL_PATH = "custom_txt_gpu_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'word_to_int': word_to_int,
    'int_to_word': int_to_word,
    'max_sequence_len': max_sequence_len,
    'total_words': total_words,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM
}, MODEL_PATH)
print(f"💾 Model successfully saved to '{MODEL_PATH}'\n")

# --- TEXT GENERATION FUNCTION ---
def generate_text(seed_text, next_words=15, temperature=0.7):
    model.eval()
    result = seed_text
    
    for _ in range(next_words):
        token_list = [word_to_int.get(w, 0) for w in result.lower().split()]
        token_list = [0] * (max_sequence_len - 1 - len(token_list)) + token_list
        token_list = token_list[-(max_sequence_len - 1):]
        
        x = torch.tensor([token_list], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output, _ = model(x, None)
            
        scaled_logits = output[0] / temperature
        probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
        predicted_idx = np.random.choice(len(probs), p=probs)
        
        output_word = int_to_word.get(predicted_idx, "")
        if output_word:
            result += " " + output_word
            
    return result

print("--- Generated PyTorch Output ---")
random_seed = np.random.choice(vocab) if vocab else "constitution"
print(f"Seed word: '{random_seed}'")
print(generate_text(random_seed, next_words=12))