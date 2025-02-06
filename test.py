import torch
import torch.nn as nn
from tokenizers import Tokenizer
from model import GRU  # Import your custom GRU model

# Hyperparameters (must match training settings)
INPUT_SIZE = 10000
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EMBEDDING_DIM = 128
SEQ_LENGTH = 200

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")  # Ensure this file exists
vocab = tokenizer.get_vocab()
pad_idx = vocab["<pad>"]

# Preprocess function (same as training)
def preprocess(text):
    token_ids = tokenizer.encode(text).ids[:SEQ_LENGTH]  # Tokenize and truncate
    return torch.tensor(token_ids + [pad_idx] * (SEQ_LENGTH - len(token_ids)))  # Pad

# Sentiment GRU Model (must match training architecture)
class SentimentGRU(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=pad_idx)
        self.gru = GRU(embedding_dim, hidden_size, num_layers)  # Use the same GRU model
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification output

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size = x.size(0)
        h_0 = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(x.device)  # Init hidden state
        output, _ = self.gru(embedded, h_0)  # Pass through GRU
        final_hidden_state = output[:, -1, :]  # Take last output
        return torch.sigmoid(self.fc(final_hidden_state))  # Binary classification

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentGRU(INPUT_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
model.load_state_dict(torch.load("sentiment_gru.pth", map_location=device))  # Load trained model
model.eval()

# Get user input for testing
while True:
    text = input("\nEnter a movie review (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break

    # Preprocess and predict
    input_tensor = preprocess(text).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # Interpret results
    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    print(f"Sentiment: {sentiment} (Score: {prediction:.4f})")
