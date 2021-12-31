import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_length:int, dropout: float = 0.1, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        div_term1 = torch.exp(torch.arange(1, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_sequence_length, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        try:
            pe[0, :, 1::2] = torch.cos(position * div_term1)
        except:
            print('ola')
            print(position.shape, div_term.shape, div_term1.shape)
        self.register_buffer('pe', pe)
        
        #add to buffer, since no backward requires
        self.register_buffer('pe', pe)
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        r"""[Add Positonal Encoding vector]

        Args:
            sequence (torch.Tensor): [shape: [batch_size, sequence_length, embedding_dim]]

        Returns:
            torch.Tensor: [shape: [batch_size, sequnce_length, embedding_dim]]
        """
        sequence = sequence + self.pe[:sequence.size(1)]
        sequence = self.dropout(sequence)
        return sequence
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 3, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.wq = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_heads)])
        self.wk = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_heads)])
        self.wv = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_heads)])
        self.softmax = nn.Softmax(dim = 2)
        self.sqrt_dim = math.sqrt(embedding_dim)
        self.w0 = nn.Sequential(
            nn.Dropout(p= dropout),
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
        )
        
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        r"""[Multihead Attention]

        Args:
            sequence (torch.Tensor): [shape: [batch_size, sequence_length, embedding_dim]]

        Returns:
            torch.Tensor: [shape: [batch_size, sequence_length, embedding_dim]]
        """
       
        Qs = torch.stack([wq(sequence) for wq in self.wq])
        Ks = torch.stack([wk(sequence) for wk in self.wk])
        Vs = torch.stack([wv(sequence) for wv in self.wv])
        
        #perform dot product to calculate attention
        value_tensor = torch.matmul(Qs, torch.transpose(Ks, 2, 3))
        
        #standardize by performing softmax
        value_tensor = self.softmax(value_tensor)
        #calculate output
        value_tensor = torch.matmul(value_tensor, Vs) / self.sqrt_dim
        
        value_tensor = torch.split(value_tensor, 1)
        value_tensor = torch.cat([tensor.squeeze(0) for tensor in value_tensor], dim = 2)
        
        #combine output of heads
        return self.w0(value_tensor)
