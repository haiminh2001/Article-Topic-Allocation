
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
import time

dir_path = dirname(dirname(abspath(__file__)))

        

class Encoder(nn.Module):
    def __init__(self, max_vocab_length: int, num_heads = 3, sequence_length: int = 4, embedding_dim: int = 100, dropout: float = 0.1, hide_target_rate: float = 0.5  ,**kwargs):
        super(Encoder, self).__init__()
        buffer = max(int((max_vocab_length + embedding_dim) / 20), 1000)
        self.contexts_dim_reduction = nn.Sequential(
            nn.Linear(max_vocab_length, buffer),
            nn.ReLU(),
            nn.Dropout(p= dropout),
            nn.Linear(buffer, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p= dropout),
        )
        self.targets_dim_reduction = nn.Sequential(
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
        self.hide_target_rate = hide_target_rate
        self.save_hide_target_rate = hide_target_rate
        
    def eval_mode(self):
        self.hide_target_rate = 0
    
    def train_mode(self):
        self.hide_target_rate = self.save_hide_target_rate
    
    @property
    def dim_params(self):
        dim_reduction = sum(p.numel() for p in self.contexts_dim_reduction.parameters() if p.requires_grad) + sum(p.numel() for p in self.targets_dim_reduction.parameters() if p.requires_grad)
        return dim_reduction
        
    def one_hot_dim_reduction(self, one_hot: torch.Tensor):
        return self.dim_reduction(one_hot)
    
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """[Encode words into vectors]

        Args:
            x (torch.Tensor): [sequence of words, shape: [num_sequences, max_vocab_length]]
            x0 (torch.Tensor): [context words, shape: [num_sequences, sequence_length, max_vocab_length]]

        Returns:
            torch.Tensor: [sequence of vectors, shape: [num_sequences, embedding_dim]]
        """
        #dim reduction
        x01 = self.contexts_dim_reduction(x0) 
        
        #hide target or not
        hide = torch.rand(1)[0]
        if (hide < self.hide_target_rate):
            x1 = torch.zeros(x.shape[0], x01.shape[-1]).cuda()
        else:
            x1 = self.targets_dim_reduction(x)
        
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
        buffer1 = min(embedding_dim * 5, 1000)
        buffer2 = int((buffer1 * 5 + max_vocab_length)  / 6)
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
    def __init__(self, max_vocab_length:int, embedding_dim: int = 200, num_heads:int = 3, window_size: int = 4, dropout: float= 0.1, lr: float= 1e-4, eps: float= 1e-5, hide_target_rate: float = 0.5, **kwargs):
        super(WordEmbeddingModel, self).__init__()
        self.lr = lr
        self.eps = eps
        self.encode = Encoder(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, sequence_length= 2 * window_size + 1, dropout= dropout, hide_target_rate = hide_target_rate)
        self.decode = Decoder(embedding_dim= embedding_dim, sequence_length= 2 * window_size + 1, dropout= dropout, max_vocab_length= max_vocab_length)
        self.max_vocab_length = max_vocab_length
    
    def eval_mode(self):
        self.encode.eval_mode()
        
    def train_mode(self):
        self.encode.train_mode()
    
    def forward(self, x: torch.Tensor, x0:torch.Tensor):
        out = self.encode(x, x0)
        out = self.decode(out)
        return out
    
    def embed(self, x: torch.Tensor, x0: torch.Tensor):
        return self.encode(x, x0)
    
    def one_hot_dim_reduction(self, one_hot: torch.Tensor):
        return self.encode.one_hot_dim_reduction(one_hot= one_hot)
    
    def training_step(self, batch, batch_idx):
        contexts, targets = batch
        contexts = F.one_hot(contexts, self.max_vocab_length).type(torch.float).squeeze()
        targets = F.one_hot(targets, self.max_vocab_length).type(torch.float).squeeze()
        out = self.encode(targets, contexts)
        out = self.decode(out)
        #cross entropy since out put is in one hot form
        loss = F.cross_entropy(out, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print('Epochs {}: loss: {}'.format(self.current_epoch, avg_loss))
        
    @property
    def num_params(self):
        dim_params = self.encode.dim_params
        encode_params = sum(p.numel() for p in self.encode.parameters() if p.requires_grad) - dim_params
        decode_params = sum(p.numel() for p in self.decode.parameters() if p.requires_grad)
        total_params = encode_params + decode_params + dim_params
        return dim_params, encode_params, decode_params, total_params
    
    def configure_optimizers(self):
        
        optimizer_grouped_parameters = [
            {
                "params": p
                    for p in self.encode.parameters()
            },
            {
                "params": p
                    for p in self.decode.parameters()
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
        model_file: str = '/data_module/word_embedder.ckpt', 
        gpus: int = 1,
        hide_target_rate: float = 0.5,
        ):
        self.window_size = window_size
        self.vocab_builder = vocab_builder
        self.model_file = model_file
        self.max_vocab_length = max_vocab_length
        try:
            self.gpus = gpus
        except:
            print('Require at least 1 gpu!')
            return
            
        if load_embedder:
            try:
                self.load()
            except:
                print('No embedder found')
                self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps, hide_target_rate= hide_target_rate)
        else:
            self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps, hide_target_rate= hide_target_rate)

        print(self)
        
        
    def __str__(self) -> str:
        dim_params, encode_params, decode_params, total = self.model.num_params
        info = 'Weights summary\n==========================================\n'
        info += f'Dimension reduction: {(dim_params / 1e6):.1f} M\n'
        info += f'Encoder: {(encode_params / 1e6):.1f} M\n'
        info += f'Decoder: {(decode_params / 1e6):.1f} M\n'
        info += '==========================================\n'
        info += f'Total: {(total /1e6):.1f} M\n'
        info += f'Actual params used for embedding: {(encode_params / 1e6):.1f} M'
        return info
    
        
    def setup_trainer(self, gpus, epochs):
        self.trainer = Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 5)
    
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
        #turn on train mode
        self.model.train_mode()
        self.setup_trainer(gpus= self.gpus, epochs = epochs)
        for i in range(dataset_splits):
            s = time.time()
            #prepare data
            self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, dataset_splits= dataset_splits, split_index= self.count)
            #fit        
            self.trainer.fit(
                model= self.model,
                train_dataloaders= self.data_loader,
            )
            t = time.time()
            print(f'Finished dataset {i + 1}, total time: {t - s}, time per epoch: {(t - s) / epochs}')
            del self.data_loader
            self.save()
            
    def one_hot_dim_reduction(self, one_hot: torch.Tensor):
        return self.model.one_hot_dim_reduction(one_hot= one_hot)
    
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
        
        #turn on eval mode
        self.model.eval_mode()
        
        #embed
        print('Embedding')
        for contexts, targets in tqdm(self.data_loader):
            #words is a matrix representing a bunch of words, with each row corresponds to a word
            words.append(self.model.embed(contexts, targets))
            
        #concatenate into a tensor
        words = torch.cat(self.words)
        
        #wrap in a dataset
        return ClassifierInputDataset(words, self.text_ends)
        
        
    