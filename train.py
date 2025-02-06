import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset
from model import GRU  # Import your custom GRU model

# Set random seed
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
INPUT_SIZE = 10000  # Vocabulary size
HIDDEN_SIZE = 128   # GRU hidden units
NUM_LAYERS = 2      # Number of GRU layers
EMBEDDING_DIM = 128 # Word embedding size
BATCH_SIZE = 32
EPOCHS = 5
SEQ_LENGTH = 200  # Max tokens per review
LEARNING_RATE = 0.001

# ==========================
# 1. Load and Preprocess Data
# ==========================

# Load IMDb dataset from CSV
df = pd.read_csv("imdb_small.csv")  # Ensure this file exists
reviews = df["review"].tolist()
labels = df["sentiment"].tolist()

# Convert labels to binary (1 = positive, 0 = negative)
labels = [1 if label == "positive" else 0 for label in labels]

# Initialize BPE Tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Train BPE Tokenizer
trainer = BpeTrainer(vocab_size=INPUT_SIZE, special_tokens=["<unk>", "<pad>"])
tokenizer.train_from_iterator(reviews, trainer)

tokenizer.save("tokenizer.json") 
# Get vocabulary
vocab = tokenizer.get_vocab()
pad_idx = vocab["<pad>"]

# Convert text to tensor format
def preprocess(text):
    token_ids = tokenizer.encode(text).ids[:SEQ_LENGTH]  # Tokenize and truncate
    return torch.tensor(token_ids + [pad_idx] * (SEQ_LENGTH - len(token_ids)))  # Pad

# Custom Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = [(preprocess(text), label) for text, label in zip(texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Load dataset into DataLoader
train_dataset = IMDBDataset(reviews, labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================
# 2. Modify Your GRU Model for Classification
# ==========================
class SentimentGRU(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=pad_idx)
        self.gru = GRU(embedding_dim, hidden_size, num_layers)  # Use your custom GRU
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification output

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size = x.size(0)
        h_0 = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(x.device)  # Init hidden state
        output, _ = self.gru(embedded, h_0)  # Pass through your GRU
        final_hidden_state = output[:, -1, :]  # Take last output
        return torch.sigmoid(self.fc(final_hidden_state))  # Binary classification

# ==========================
# 3. Train Your Model with tqdm Progress Bar & Save Checkpoint
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentGRU(INPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


print (device)

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print at least 1
print(torch.cuda.get_device_name(0))  # Should print your GPU name

# Training loop with tqdm
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")  # tqdm progress bar
    
    for batch in progress_bar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))  # Update progress bar loss

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # **Save model after the last epoch**
    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), "sentiment_gru.pth")
        print("Model saved as 'sentiment_gru.pth' ✅")

print("Training complete! 🎉")
