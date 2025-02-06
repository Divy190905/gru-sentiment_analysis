import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Update gate parameters
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate hidden state parameters
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        r_t = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))  # Reset gate
        z_t = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))  # Update gate
        h_tilde = torch.tanh(self.W_h(x) + r_t * self.U_h(h_prev))  # Candidate hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde  # Final hidden state
        return h_t

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru_cells = nn.ModuleList([GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
    
    def forward(self, x, h_0):
        batch_size, seq_length, _ = x.size()
        h_t = h_0
        outputs = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            h_layer = []
            for layer in range(self.num_layers):
                h_out = self.gru_cells[layer](x_t, h_t[layer])
                h_layer.append(h_out)
                x_t = h_out
            h_t = torch.stack(h_layer)
            outputs.append(h_t[-1])
        
        outputs = torch.stack(outputs, dim=1)  # Shape (batch_size, seq_length, hidden_size)
        return outputs, h_t


