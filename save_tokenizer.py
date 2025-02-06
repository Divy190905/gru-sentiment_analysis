import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Load IMDb dataset
df = pd.read_csv("imdb_small.csv")  # Ensure this file exists
reviews = df["review"].tolist()

# Initialize and train tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>"])
tokenizer.train_from_iterator(reviews, trainer)

# Save tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer saved successfully! ✅")
