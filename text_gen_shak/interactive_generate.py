import torch
import numpy as np
import torch.nn as nn

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define model architecture (must match training)
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

# 3. Load saved checkpoint dictionary
MODEL_PATH = "shakespeare_pytorch_model.pth"
print(f"Loading model checkpoint from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

word_to_int = checkpoint['word_to_int']
int_to_word = checkpoint['int_to_word']
max_sequence_len = checkpoint['max_sequence_len']

# 4. Initialize model and load trained weights
model = WordLSTMGenerator(
    checkpoint['total_words'], 
    checkpoint['embedding_dim'], 
    checkpoint['hidden_dim']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded and ready on GPU!\n")

# 5. Interactive generation loop
def generate_text(seed_text, next_words=15, temperature=0.7):
    result = seed_text.lower()
    
    for _ in range(next_words):
        token_list = [word_to_int.get(w, 0) for w in result.split()]
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

# Run interactive prompt
if __name__ == "__main__":
    while True:
        prompt = input("\nEnter a seed prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        if not prompt.strip():
            continue
            
        generated = generate_text(prompt, next_words=15, temperature=0.7)
        print(f"Generated: {generated}")