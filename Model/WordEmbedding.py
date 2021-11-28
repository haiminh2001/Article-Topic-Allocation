import torch
from torch import nn
import pytorch_lightning as pl
from VocabularyBuilder import VocabularyBuilder
from torch.nn.functional import normalize
from TransformerLayers import PositionalEncoding, MultiHeadAttention
from pytorch_lightning import Trainer
from transformers import AdamW
from Data.DatatModule import EmbedDataset
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from os.path import dirname, abspath
parent_path = dirname(dirname(abspath(__file__)))

print(parent_path)
class Encoder(nn.Module):
    def __init__(self, max_vocab_length: int, num_heads = 3, sequence_length: int = 4, embedding_dim: int = 100, dropout: float = 0.1 ,**kwargs):
        super(Encoder, self).__init__()
        buffer = int((max_vocab_length + embedding_dim) / 2)
        self.dim_reduction = nn.Sequential(
            nn.Linear(max_vocab_length, buffer),
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
            x (torch.Tensor): [sequence of words, shape: [num_sequences, sequence_length, max_vocab_length]]

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

class Decoder(nn.Module):
    def __init__(self,embedding_dim:int, sequence_length: int = 4, kernel_size: int = 3, dropout: float = 0.1, **kwargs):
        super(Decoder, self).__init__()
        buffer1 = int(embedding_dim / 2)
        buffer2 = int(buffer1 / 2)
        
        #convo block 1
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(embedding_dim, buffer1 , kernel_size= 3, padding=1),
            nn.BatchNorm1d(buffer1),
            nn.Dropout(p= dropout),
        )
        
        #convo block 2
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(buffer1, buffer2, kernel_size= 3, padding= 1),
            nn.BatchNorm1d(buffer2),
            nn.Dropout(p= dropout),
        )
        
        #fully connected
        buffer3 = int((buffer2 + buffer1) * sequence_length / 2)
        self.fc = nn.Sequential(
            nn.Linear((buffer2 + buffer1) * sequence_length, buffer3),
            nn.Dropout(p= dropout),
            nn.Linear(buffer3, embedding_dim),
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            sequences (torch.Tensor): [shape: [num_sequences, sequence_length, embedding_size]]

        Returns:
            torch.Tensor: [shape: [num_sequences, embeding_size]]
        """
        x = torch.transpose(sequences,1,2)
        x = self.conv1d_1(x)
        x1 = self.conv1d_2(x)
        x2 = torch.cat((torch.flatten(x, start_dim= 1), torch.flatten(x1, start_dim= 1)), dim = 1)
        x2 = self.fc(x2)
        return x2
    
class WordEmbeddingModel(pl.LightningModule):
    def __init__(self, max_vocab_length:int, embedding_dim: int = 200, num_heads:int = 3, window_size: int = 4, dropout: float= 0.1, lr: float= 1e-4, eps: float= 1e-5, **kwargs):
        super(WordEmbeddingModel, self).__init__()
        self.lr = lr
        self.eps = eps
        self.encode = Encoder(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, sequence_length= 2 * window_size + 1, dropout= dropout)
        self.decode = Decoder(embedding_dim= embedding_dim, sequence_length= 2 * window_size + 1, dropout= dropout)
        
    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out
    
    def embed(self, x):
        return self.encode(x)
    
    def training_step(self, batch, batch_idx):
        contexts, targets = batch
        out = self.encode(contexts)
        out = self.decode(out)
        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.cat([output['loss'] for output in outputs]).detech()
        avg_loss = torch.mean(avg_loss)
        print('Epochs {}: loss: {}'.format(self.current_epoch, avg_loss))
    
    def configure_optimizers(self):
        encode_optimizer = AdamW(
            self.encode.parameters,
            lr= self.lr,
            eps=self.eps,
        )
        decode_optimizer = AdamW(
            self.decode.parameters,
            lr= self.lr,
            eps=self.eps,
        )
        return encode_optimizer, decode_optimizer
    
class WordEmbedder():
    def __init__(
        self, 
        vocab_builder: VocabularyBuilder = None, 
        max_vocab_length: int = 20000, 
        embedding_dim: int = 200, 
        num_heads: int = 3, 
        dropout: float = 0.1, 
        lr: float = 1e-4,
        eps: float = 1e-5, 
        load_embedder: bool = True,
        window_size: int = 3,
        model_file: str = '/Data/word_embedder.pickle', 
        gpus: int = 1,
        ):

        self.window_size = window_size
        self.vocab_builder = vocab_builder
        self.model_file = model_file
        
        self.setup_trainer(gpus)
        
        self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps)
        
        if load_embedder:
            self.load()
            
    def setup_trainer(self, gpus):
        self.trainer = Trainer(gpus = gpus)
    
    def setup_data(self, texts: list, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True):
        dataset = EmbedDataset(texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
        self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True, pin_memory= pin_memory, num_workers= num_workers)
    
    def fit(self, texts: list, epochs: int = 20, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True):
        #prepare data
        self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory)
        #fit
        self.trainer.fit(
            model= self.model,
            train_dataloaders= self.data_loader,
            max_epochs= epochs,
        )
    
    def save(self):
        torch.save(self.model, parent_path + self.model_file)
        print('Saved word embedder')
    
    def load(self):
        print('Loading word embedder')
        self.model = torch.load(parent_path + self.model_file)
    
    def embed_vocab(self):
        pass
    
    
    