import os
import time
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 1. Load Dataset
csv_filename = "final_dataset.csv"
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found!")
    exit(1)

print("Loading CSV file into memory...")
df = pd.read_csv(csv_filename).dropna(subset=['text', 'label'])
corpus = df['text'].astype(str).str.lower().tolist()
labels = df['label'].tolist()
print(f"Loaded {len(corpus)} rows successfully!")

# Encode labels to integers
unique_labels = sorted(list(set(labels)))
label_to_int = {l: i for i, l in enumerate(unique_labels)}
y = torch.tensor([label_to_int[l] for l in labels], dtype=torch.long)
num_classes = len(unique_labels)

# 2. Tokenization & Padding
print("Building vocabulary and padding sequences...")
words = " ".join(corpus).split()
vocab = sorted(list(set(words)))
word_to_int = {w: i for i, w in enumerate(vocab)}
total_words = len(vocab)
print(f"Total unique words in vocabulary: {total_words}")

MAX_SEQ_LEN = 50
padded_sequences = []
for line in corpus:
    token_list = [word_to_int[w] for w in line.split() if w in word_to_int]
    padded = [0] * (MAX_SEQ_LEN - len(token_list)) + token_list[-MAX_SEQ_LEN:]
    padded_sequences.append(padded)

X = torch.tensor(padded_sequences, dtype=torch.long)

# 3. Classifier Model Architecture
class SecurityClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SecurityClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        out, (hn, cn) = self.lstm(embed)
        out = self.fc(hn[-1])
        return out

model = SecurityClassifier(total_words, embedding_dim=64, hidden_dim=128, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 4. Training Loop with Live Progress
EPOCHS = 6
BATCH_SIZE = 64
dataset_size = len(X)
total_batches = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Starting training for {EPOCHS} epochs ({total_batches} batches per epoch)...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    permutation = torch.randperm(dataset_size)
    start_time = time.time()
    
    for batch_idx, i in enumerate(range(0, dataset_size, BATCH_SIZE)):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X[indices].to(device), y[indices].to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
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

# Save Classifier
torch.save({
    'model_state_dict': model.state_dict(), 
    'word_to_int': word_to_int, 
    'label_to_int': label_to_int
}, "security_classifier.pth")
print("Classifier model saved successfully to 'security_classifier.pth'!")