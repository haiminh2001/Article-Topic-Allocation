import torch
from torch import nn
import pytorch_lightning as pl
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_sequence_length:int, dropout: float = 0.1, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        #create postion vector and add unsqueeze to perform vector product later on
        position = torch.arange(max_sequence_length)
        
        #create post_term represent the product inside cosine, step = 2 because of dividing into 2 parts: 2i and 2i + 1
        self.exponential = -math.log(10000) / dim
        pos_term = torch.exp(torch.arange(0, dim, 2) * self.exponential)
        
        #pe shape [1, max_sequence_length, embedding_dim]
        pe = torch.zeros(max_sequence_length, dim)
        pe[:, 0::2] = torch.sin(pos_term * position)
        pe[:, 1::2] = torch.cos(pos_term * position)
        
        #add to buffer, since no backward requires
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape [batch_size, seq_len, embedding_dim]

        Returns:
            torch.Tensor
        """
        
        x = x + self.pe[:x.size(1)]
        x = self.dropout(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, n: int):
        super(SelfAttention, self).__init__()
        