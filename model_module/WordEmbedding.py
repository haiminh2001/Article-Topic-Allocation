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
from torchsummary import summary

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
        self.combine = nn.Sequential(
            nn.Linear(sequence_length, 1),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
    
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """[Encode words into vectors]

        Args:
            x (torch.Tensor): [sequence of words, shape: [num_sequences, max_vocab_length]]
            x0 (torch.Tensor): [context words, shape: [num_sequences, sequence_length, max_vocab_length]]

        Returns:
            torch.Tensor: [sequence of vectors, shape: [num_sequences, embedding_dim]]
        """
        #dim reduction
        x1 = self.dim_reduction(x)
        x01 = self.dim_reduction(x0) 
        
        #add positional encoding
        x01 = self.pe(x01)

        #multihead attention
        z1 = self.mha(x01)
        
        #combine all context words into 1 vector
        z1 = self.combine(torch.transpose(z1, 1, 2)).squeeze()

        #add and normalize
        z1 = normalize(z1 + x1, dim= 1)
        
        #feadforwad
        z2 = self.fc(z1)
        
        #add and normalize
        z2 = normalize(z2 + z1 + x1, dim= 1)
        
        return z2

class Decoder(nn.Module):
    def __init__(self,max_vocab_length:int, embedding_dim:int, dropout: float = 0.1, **kwargs):
        super(Decoder, self).__init__()
        buffer1 = embedding_dim * 2
        buffer2 = int((buffer1 + max_vocab_length)  / 2)
        #fc
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, buffer1),
            nn.ReLU(),
            nn.Dropout(p= dropout),
            nn.Linear(buffer1, buffer2),
            nn.ReLU(),
            nn.Dropout(p= dropout),
            nn.Linear(buffer2, max_vocab_length),
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
        
        return self.fc(encoded)
    
class WordEmbeddingModel(pl.LightningModule):
    def __init__(self, max_vocab_length:int, embedding_dim: int = 200, num_heads:int = 3, window_size: int = 4, dropout: float= 0.1, lr: float= 1e-4, eps: float= 1e-5, **kwargs):
        super(WordEmbeddingModel, self).__init__()
        self.lr = lr
        self.eps = eps
        self.encode = Encoder(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, sequence_length= 2 * window_size + 1, dropout= dropout)
        self.decode = Decoder(embedding_dim= embedding_dim, sequence_length= 2 * window_size + 1, dropout= dropout, max_vocab_length= max_vocab_length)
        
    def forward(self, x: torch.Tensor, x0:torch.Tensor):
        out = self.encode(x, x0)
        out = self.decode(out)
        return out
    
    def embed(self, x: torch.Tensor, x0: torch.Tensor):
        return self.encode(x, x0)
    
    def training_step(self, batch, batch_idx):
        contexts, targets = batch
        out = self.encode(targets, contexts)
        out = self.decode(out)
        #cross entropy since out put is in one hot form
        loss = F.cross_entropy(out, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print('Epochs {}: loss: {}'.format(self.current_epoch, avg_loss))
    
    def configure_optimizers(self):
        
        optimizer_grouped_parameters = [
            {
                "params": p
                    for n, p in self.encode.named_parameters()
            },
            {
                "params": p
                    for n, p in self.decode.named_parameters()
            },
        ]
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr= self.lr,
            eps= self.eps,
        )
        return optimizer
    
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
        self.window_size = window_size
        self.vocab_builder = vocab_builder
        self.model_file = model_file
        self.max_vocab_length = max_vocab_length
        self.gpus = gpus
        if load_embedder:
            try:
                self.load()
            except:
                print('No embedder found')
                self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps)
        else:
            self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps)
            
    def setup_trainer(self, gpus, epochs):
        self.trainer = Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None)
    
    def setup_data(self, split_index: int, texts: list, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, inference = False, dataset_splits: int = 10):
        self.count +=1
        if inference:
            dataset = InferenceDataset(split_index= split_index, dataset_splits = dataset_splits, texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
            self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False, pin_memory= pin_memory, num_workers= num_workers)
            self.text_ends = dataset.get_text_ends
        else:
            dataset = EmbedDataset(split_index= split_index, dataset_splits= dataset_splits, texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
            self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True, pin_memory= pin_memory, num_workers= num_workers)
        
        if self.count == dataset_splits:
            del dataset
    
    def fit(self, texts: list, epochs: int = 20, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, 
            dataset_splits: int = 10):
        self.count = 0
        self.setup_trainer(gpus= self.gpus, epochs = epochs)
        for _ in range(dataset_splits):
            #prepare data
            self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, dataset_splits= dataset_splits, split_index= self.count)
            #fit        
            self.trainer.fit(
                model= self.model,
                train_dataloaders= self.data_loader,
            )
            del self.data_loader
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
        self.count = 0
        self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, inference= True, split_index= self.count)
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
        
        
    