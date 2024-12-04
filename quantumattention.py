import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class QuantumAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def collapse_fn(self, superposition, threshold=1e-6):
        # Simulate quantum collapse based on probability amplitudes
        probs = F.softmax(superposition, dim=-1)
        
        # Apply threshold for collapse
        mask = (probs > threshold).float()
        collapsed = probs * mask
        
        # Renormalize
        collapsed = collapsed / (collapsed.sum(dim=-1, keepdim=True) + 1e-9)
        return collapsed
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores (superposition state)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply quantum collapse instead of standard softmax
        attention = self.collapse_fn(scores)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


if __name__ == '__main__':
  # Initialize model
  model = QuantumAttention(d_model=512, num_heads=8)
  
  # Create sample input
  batch_size = 32
  seq_length = 50
  x = torch.randn(batch_size, seq_length, 512)

  # Forward pass
  output = model(x)
