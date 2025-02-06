# GRU Sentiment Analysis from Scratch  

## Overview  
This project implements a **GRU (Gated Recurrent Unit) model from scratch** for **sentiment analysis** on the IMDb dataset. The entire model is built from the ground up, including the GRU cell, training pipeline, and tokenization.  

## Key Features  
- **GRU Implemented from Scratch** – No external libraries for GRU layers were used.  
- **Trained Completely from Scratch** – Starting with random weights, the model was trained efficiently.  
- **Loss Reduced from 4.2 to 0.151** – Achieved significant improvement in training loss.  
- **Byte Pair Encoding (BPE) Tokenizer** – Used for efficient text representation.  
- **Batch Training with DataLoader** – Improved training efficiency with mini-batches.  
- **TQDM Progress Bar** – Added for better tracking of training progress.  

## Training Process  
- The model was trained on the **full IMDb dataset** (`imdb_small.csv`).  
- Used **Binary Cross-Entropy Loss (BCELoss)** for classification.  
- Optimized using **Adam Optimizer** with a learning rate of `0.001`.  
- Training was run for **5 epochs** with a **batch size of 32**.  

## Model Performance  
- Achieved a final **training loss of 0.151** after starting at **4.2**.  
- The model successfully classifies reviews into **positive** and **negative** sentiments.  

## Tokenization  
- Implemented **Byte Pair Encoding (BPE)** for tokenization.  
- Vocabulary size: **10,000** words.  
- Maximum sequence length: **200 tokens** per review.  

## Testing  
- The model was tested on **custom inputs** as well as the full dataset.  
- Includes **a script for batch testing** with a progress bar (`tqdm`).  

## Example Predictions  
*(Include example model outputs here if needed)*  

## Model Architecture  
*(Include a brief explanation of your custom GRU model architecture here if needed)*  

## Output Screenshots  
Below are the output images showcasing the model's predictions and performance:  

![Model Output](output1.png)  
![Loss Reduction](output2.png)  


