import torch
from torch import nn
import pytorch_lightning as pl
from data_module import VocabBuilder, EmbedDataset, InferenceDataset, ClassifierInputDataset
from torch.nn.functional import normalize
from .TransformerLayers import PositionalEncoding, MultiHeadAttention
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.nn import functional as F 
from torch.utils.data import DataLoader, Dataset
from os.path import dirname, abspath
from tqdm import tqdm
dir_path = dirname(dirname(abspath(__file__)))

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
    
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """[Encode words into vectors]

        Args:
            x (torch.Tensor): [sequence of words, shape: [num_sequences, max_vocab_length]]
            x0 (torch.Tensor): [context words, shape: [num_sequences, max_vocab_length]]

        Returns:
            torch.Tensor: [sequence of vectors, shape: [num_sequences, sequence_length, embedding_dim]]
        """
        #dim reduction
        x1 = self.dim_reduction(x)
        x01 = self.dim_reduction(x0) 
        
        #add positional encoding
        x1 = self.pe(x1)
        
        #multihead attention
        z1 = self.mha(x1)

        #add and normalize
        z1 = normalize(z1 + x1, dim= 1)
        
        #feadforwad
        z2 = self.fc(z1)
        
        #add and normalize
        z2 = normalize(z2 + z1 + x01, dim= 1)
        
        return z2

class Decoder(nn.Module):
    def __init__(self,max_vocab_length:int, embedding_dim:int, dropout: float = 0.1, **kwargs):
        super(Decoder, self).__init__()
        buffer1 = embedding_dim * 2
        buffer2 = buffer1 * 2
        
        #convo block 1
        self.fc_1 = nn.Sequential(
            nn.Linear(embedding_dim, buffer1),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
        
        #convo block 2
        self.fc_2 = nn.Sequential(
            nn.Linear(buffer1, buffer2),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
        
        #fully connected
        buffer3 = int((buffer2 + max_vocab_length)  / 2)
        self.fc_3 = nn.Sequential(
            nn.Linear(buffer3, max_vocab_length),
            nn.ReLU(),
            nn.Dropout(p= dropout),            
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            sequences (torch.Tensor): [shape: [num_sequences, embedding_size]]

        Returns:
            torch.Tensor: [shape: [num_sequences, max_vocab_length]]
        """
        out = self.fc_1(encoded)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out
    
class WordEmbeddingModel(pl.LightningModule):
    def __init__(self, max_vocab_length:int, embedding_dim: int = 200, num_heads:int = 3, window_size: int = 4, dropout: float= 0.1, lr: float= 1e-4, eps: float= 1e-5, **kwargs):
        super(WordEmbeddingModel, self).__init__()
        self.lr = lr
        self.eps = eps
        self.encode = Encoder(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, sequence_length= 2 * window_size + 1, dropout= dropout)
        self.decode = Decoder(embedding_dim= embedding_dim, sequence_length= 2 * window_size + 1, dropout= dropout, max_vocab_length= max_vocab_length)
        
    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out
    
    def embed(self, x: torch.Tensor, x0: torch.Tensor):
        return self.encode(x, x0)
    
    def training_step(self, batch, batch_idx):
        contexts, targets = batch
        out = self.encode(contexts, targets)
        out = self.decode(out)
        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.cat([output['loss'] for output in outputs]).detech()
        avg_loss = torch.mean(avg_loss)
        print('Epochs {}: loss: {}'.format(self.current_epoch, avg_loss))
    
    def configure_optimizers(self):
        encode_optimizer = Adam(
            self.encode.parameters,
            lr= self.lr,
            eps=self.eps,
        )
        decode_optimizer = Adam(
            self.decode.parameters,
            lr= self.lr,
            eps=self.eps,
        )
        return encode_optimizer, decode_optimizer
    
class WordEmbedder():
    def __init__(
        self, 
        vocab_builder: VocabBuilder = None, 
        max_vocab_length: int = 20000, 
        embedding_dim: int = 200, 
        num_heads: int = 3, 
        dropout: float = 0.1, 
        lr: float = 1e-4,
        eps: float = 1e-5, 
        load_embedder: bool = True,
        window_size: int = 3,
        model_file: str = '/data_module/word_embedder.cpkt', 
        gpus: int = 1,
        ):
        print('Setting up model...')
        self.window_size = window_size
        self.vocab_builder = vocab_builder
        self.model_file = model_file
        self.max_vocab_length = max_vocab_length
        self.setup_trainer(gpus)
        if load_embedder:
            try:
                self.load()
            except:
                print('No embedder found')
                self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps)
        else:
            self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps)
            
    def setup_trainer(self, gpus):
        self.trainer = Trainer(gpus = gpus, default_root_dir= dir_path + self.model_file)
    
    def setup_data(self, texts: list, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, inference = False):
        if inference:
            dataset = InferenceDataset(texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
            self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False, pin_memory= pin_memory, num_workers= num_workers)
            self.text_ends = dataset.get_text_ends
        else:
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
        self.save()
    
    def load_vocab_builder(self, vocab_builder: VocabBuilder):
        self.vocab_builder = vocab_builder
    
    def save(self):
        self.trainer.save_checkpoint(dir_path + self.model_file)
        print('Saved word embedder')
    
    def load(self):
        print('Loading word embedder')
        self.model =  WordEmbeddingModel.load_from_checkpoint(dir_path + self.model_file)
        
    def embed(self, texts: list, batch_size: int = 512, num_workers: int = 4, pin_memory: bool = True) -> Dataset:
        """[embed input texts]

        Args:
            texts (list): [list of raw texts]

        Returns:
            torch.Tensor: [shape: [num_texts, num_sequences, embedding_dim]]
        """
        
        #prepare data
        self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, inference= True)
        words = []
        
        #embed
        print('Embedding')
        for contexts, targets in tqdm(self.data_loader):
            #words is a matrix representing a bunch of words, with each row corresponds to a word
            words.append(self.model.embed(contexts, targets))
            
        #concatenate into a tensor
        words = torch.cat(self.words)
        
        #wrap in a dataset
        return ClassifierInputDataset(words, self.text_ends)
        
        
    