## Model Architecture

The sentiment analysis model is based on a **GRU (Gated Recurrent Unit)** architecture, which is an efficient variant of Recurrent Neural Networks (RNNs) for processing sequential data. The model consists of the following components:

### GRU Cell

The **GRUCell** is implemented from scratch to form the core building block of the model. The GRU cell consists of three primary gates:

1. **Update Gate (z_t)**: Controls the degree to which the previous hidden state is retained.
2. **Reset Gate (r_t)**: Determines how much of the previous hidden state should be forgotten.
3. **Candidate Hidden State (h_tilde)**: A proposed new hidden state, adjusted by the reset gate.
   
The final hidden state, \( h_t \), is a combination of the previous hidden state and the candidate hidden state, modulated by the update gate. The mathematical formulation for each step is as follows:

- \( r_t = \sigma(W_r x + U_r h_{t-1}) \)
- \( z_t = \sigma(W_z x + U_z h_{t-1}) \)
- \( \tilde{h}_t = \tanh(W_h x + r_t * U_h h_{t-1}) \)
- \( h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t \)

Where:
- \( \sigma \) is the sigmoid function.
- \( \tanh \) is the hyperbolic tangent function.
- \( W \) and \( U \) are weight matrices for the input and previous hidden state, respectively.

### GRU Model

The **GRU** model is composed of multiple GRU cells stacked in layers. The architecture works as follows:

- The model accepts a sequence of tokenized text data as input, where each token is represented by an embedding.
- For each time step, the model processes the input through multiple GRU layers sequentially.
- The output of the last GRU layer at each time step is collected and passed to the final output layer for sentiment classification.

The forward pass through the model can be described as:

1. **Input Layer**: A sequence of tokenized text is passed into the model.
2. **GRU Layers**: The sequence is processed through the stacked GRU layers.
3. **Final Hidden State**: The final hidden state from the last GRU layer is used as a feature representation for sentiment classification.
4. **Output Layer**: The model produces a prediction, which is a probability of the review being positive or negative, using **Binary Cross-Entropy Loss (BCELoss)**.
