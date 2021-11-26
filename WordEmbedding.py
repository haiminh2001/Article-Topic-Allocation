import torch
from torch import nn
import pytorch_lightning as pl
from TransformerLayers import PositionalEncoding, SelfAttention
class WordEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(WordEmbedding, self).__init__()
        self.pe = PositionalEncoding() 
        self.sa = SelfAttention()
        


