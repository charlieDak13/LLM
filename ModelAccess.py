import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random

# ----------------------------
# 1. Download and Combine Texts
# ----------------------------

file_path = "shakespeare.txt"
file_path2 = "sonnets.txt"
url1 = "https://raw.githubusercontent.com/charlieDak13/LLM/refs/heads/master/shakespeare.txt"
url2 = "https://raw.githubusercontent.com/charlieDak13/LLM/refs/heads/master/sonnets.txt"

def fetch_file(path, url):
    if not os.path.exists(path):
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    return data

text1 = fetch_file(file_path, url1)
text2 = fetch_file(file_path2, url2)
full_text = text1 + "\n" + text2

# ----------------------------
# 2. Preprocessing
# ----------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

tokens = tokenize(full_text)
vocab = {word: i+1 for i, word in enumerate(set(tokens))}  # start from 1; reserve 0 for padding
inv_vocab = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab) + 1

encoded = [vocab[word] for word in tokens]

# Create sequence chunks
SEQ_LEN = 30
X = []
y = []

for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i+SEQ_LEN])
    y.append(encoded[i+SEQ_LEN])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# ----------------------------
# 3. Dataset and Dataloader
# ----------------------------

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TextDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# ----------------------------
# 4. Model Definition
# ----------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = LSTMClassifier(vocab_size, embed_size=128, hidden_size=256)
model.eval()

# ----------------------------
# 5. Inference with Sampling
# ----------------------------

def generate_text(start_text, max_words=30, temperature=1.0):
    tokens = tokenize(start_text)
    input_ids = [vocab.get(word, 0) for word in tokens]  # 0 = unknown/pad

    if len(input_ids) < SEQ_LEN:
        input_ids = [0] * (SEQ_LEN - len(input_ids)) + input_ids
    else:
        input_ids = input_ids[-SEQ_LEN:]

    generated_ids = input_ids.copy()

    for _ in range(max_words):
        input_tensor = torch.tensor([generated_ids[-SEQ_LEN:]], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits.squeeze(0) / temperature  # scale logits by temperature
            probabilities = F.softmax(logits, dim=-1)

            predicted_token = torch.multinomial(probabilities, num_samples=1).item()

        generated_ids.append(predicted_token)

    generated_words = [inv_vocab.get(idx, "<unk>") for idx in generated_ids]
    return " ".join(generated_words)

# ----------------------------
# Example
# ----------------------------

start_prompt = "Romeo"
generated = generate_text(start_prompt, max_words=30, temperature=0.8)

print("Input Prompt:")
print(start_prompt)
print("\nGenerated Continuation:")
print(generated)
