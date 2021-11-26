import torch
from torch import nn
import pytorch_lightning as pl
from torch.functional import norm
from torch.nn.functional import normalize
from TransformerLayers import PositionalEncoding, MultiHeadAttention
class Encoder(nn.Module):
    def __init__(self, vocab_length: int, num_heads = 3, sequence_length: int = 4, embedding_dim: int = 100, dropout: float = 0.1 ,**kwargs):
        super(Encoder, self).__init__()
        buffer = int((vocab_length + embedding_dim) / 2)
        self.dim_reduction = nn.Sequential(
            nn.Linear(vocab_length, buffer),
            nn.ReLU(),
            nn.Dropout(p= dropout),
            nn.Linear(buffer, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
        self.pe = PositionalEncoding(embedding_dim, sequence_length) 
        self.mha = MultiHeadAttention(embedding_dim, num_heads= num_heads)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[Encode words into vectors]

        Args:
            x (torch.Tensor): [sequence of words, shape: [num_sequences, sequence_length, vocab_length]]

        Returns:
            torch.Tensor: [sequence of vectors, shape: [num_sequences, sequence_length, embedding_dim]]
        """
        #dim reduction
        x1 = self.dim_reduction(x)
        
        #add positional encoding
        x1 = self.pe(x1)
        
        #multihead attention
        z1 = self.mha(x1)

        #add and normalize
        z1 = normalize(z1 + x1, dim= 2)
        
        #feadforwad
        z2 = self.fc(z1)
        
        #add and normalize
        z2 = normalize(z2 + z1, dim= 2)
        
        return z2


