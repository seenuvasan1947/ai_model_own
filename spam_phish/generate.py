import os
import time
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 1. Load Dataset Safely
csv_filename = "final_dataset.csv"
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found!")
    exit(1)

print("Loading full dataset into memory...")
df = pd.read_csv(csv_filename).dropna(subset=['text'])
corpus = df['text'].astype(str).str.lower().tolist()
print(f"Loaded {len(corpus)} records successfully!")

# 2. Build Vocabulary from Full Corpus
print("Building global vocabulary dictionary...")
words = " ".join(corpus).split()
vocab = sorted(list(set(words)))
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for i, w in enumerate(vocab)}
total_words = len(vocab)
print(f"Total vocabulary size: {total_words}")

MAX_SEQ_LEN = 30  # Fixed window length per sample

# 3. Memory-Efficient Custom PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, corpus, word_to_int, max_seq_len):
        self.samples = []
        print("Tokenizing and structuring sequences into memory-safe tensors...")
        
        for line in corpus:
            token_list = [word_to_int.get(w, 0) for w in line.split() if w in word_to_int]
            if len(token_list) > 1:
                # Create rolling windows within each line to feed the generator efficiently
                for i in range(1, len(token_list)):
                    seq = token_list[:i+1]
                    padded = [0] * (max_seq_len - len(seq)) + seq[-max_seq_len:]
                    self.samples.append(padded)
                    
        # Convert to a single compact tensor array
        self.data = torch.tensor(self.samples, dtype=torch.long)
        print(f"Total processed sequence samples ready: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[-1]

dataset = TextDataset(corpus, word_to_int, MAX_SEQ_LEN)

# Use DataLoader with batching to stream data dynamically without OOM crashes
BATCH_SIZE = 256
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# 4. LSTM Generator Model Architecture
class SpamLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SpamLSTMGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

model = SpamLSTMGenerator(total_words, embedding_dim=100, hidden_dim=150).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 5. Training Loop with Live Batch Progress Bar
EPOCHS = 5
total_batches = len(dataloader)

print(f"Starting generator training for {EPOCHS} epochs ({total_batches} batches per epoch)...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output, _ = model(batch_x, None)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Live progress counter
        percent = (batch_idx + 1) / total_batches * 100
        sys.stdout.write(f"\rEpoch [{epoch+1}/{EPOCHS}] Progress: {percent:5.1f}% | Batch {batch_idx+1}/{total_batches}")
        sys.stdout.flush()
        
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_batches
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Completed | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

# Save Generator Model
torch.save({
    'model_state_dict': model.state_dict(),
    'word_to_int': word_to_int,
    'int_to_word': int_to_word,
    'max_sequence_len': MAX_SEQ_LEN
}, "spam_generator.pth")
print("Spam Generator model successfully saved to 'spam_generator.pth'!")

# 6. Quick Test Generation Function
def generate_spam(seed_text, next_words=12, temperature=0.7):
    model.eval()
    result = seed_text
    for _ in range(next_words):
        token_list = [word_to_int.get(w, 0) for w in result.lower().split()]
        token_list = [0] * (MAX_SEQ_LEN - 1 - len(token_list)) + token_list
        token_list = token_list[-(MAX_SEQ_LEN - 1):]
        
        x = torch.tensor([token_list], dtype=torch.long).to(device)
        with torch.no_grad():
            output, _ = model(x, None)
            
        scaled_logits = output[0] / temperature
        probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
        predicted_idx = np.random.choice(len(probs), p=probs) if len(probs) > 0 else 0
        
        output_word = int_to_word.get(predicted_idx, "")
        if output_word:
            result += " " + output_word
    return result

import numpy as np
print("\n--- Generated Synthetic Phishing/Spam Sample ---")
print(generate_spam("urgent", next_words=12))