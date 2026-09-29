import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Saved Model & Vocabulary
print("Loading saved security classifier model...")
checkpoint = torch.load("security_classifier.pth", map_location=device)

word_to_int = checkpoint['word_to_int']
label_to_int = checkpoint['label_to_int']
# Reverse the label dictionary so we can map numbers back to text names (e.g., 0 -> ham)
int_to_label = {i: l for l, i in label_to_int.items()}

total_words = len(word_to_int)
num_classes = len(label_to_int)
MAX_SEQ_LEN = 50

# 2. Re-define the Model Architecture
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
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully on GPU and ready for testing!\n")

# 3. Prediction Function
# def classify_message(text):
#     # Tokenize and pad the input text
#     token_list = [word_to_int.get(w, 0) for w in text.lower().split() if w in word_to_int]
#     padded = [0] * (MAX_SEQ_LEN - len(token_list)) + token_list[-MAX_SEQ_LEN:]
    
#     tensor_x = torch.tensor([padded], dtype=torch.long).to(device)
    
#     with torch.no_grad():
#         output = model(tensor_x)
#         probabilities = torch.softmax(output, dim=1)
#         predicted_idx = torch.argmax(probabilities, dim=1).item()
#         confidence = probabilities[0][predicted_idx].item() * 100
        
#     predicted_label = int_to_label.get(predicted_idx, "Unknown")
#     return predicted_label, confidence
def classify_message(text):
    # Tokenize and pad the input text
    token_list = [word_to_int.get(w, 0) for w in text.lower().split() if w in word_to_int]
    padded = [0] * (MAX_SEQ_LEN - len(token_list)) + token_list[-MAX_SEQ_LEN:]
    
    tensor_x = torch.tensor([padded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor_x)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100
        
    predicted_label = int_to_label.get(predicted_idx, "Unknown")
    
    # Ensure it's treated as a string to prevent formatting errors
    return str(predicted_label), confidence

# 4. Interactive Testing Loop
print("--- Security Classifier Inference Shell ---")
print("Type a message to check if it's Ham, Spam, or Phishing (type 'exit' to quit):")

while True:
    user_input = input("\nEnter message: ")
    if user_input.lower() == 'exit':
        break
    if not user_input.strip():
        continue
        
    label, conf = classify_message(user_input)
    print(f"👉 Prediction: **{label.upper()}** (Confidence: {conf:.2f}%)")