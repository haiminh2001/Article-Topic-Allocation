

import torch
from torch import nn
import pytorch_lightning as pl
from data_module import VocabBuilder, EmbedDataset, InferenceDataset
from .TransformerLayers import PositionalEncoding, MultiHeadAttention
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from os.path import dirname, abspath
import time
import pickle
import os
import glob
    

def remove_file_in_folders(folder_path:str, spare:str = 'train'):
    files = glob.glob(folder_path + '/*')
    pattern = spare + '_tensor'
    for f in files:
        if pattern in f:
            continue
        os.remove(f)

dir_path = dirname(dirname(abspath(__file__)))
hprams_file= '/data_module/saved_data/word_embedder_hprams.pickle'
model_file= '/data_module/saved_data/word_embedder.ckpt'
train_info_file = '/data_module/saved_data/embed_train_info.pickle'
test_info_file = '/data_module/saved_data/embed_test_info.pickle'
tensors_folder = '/data_module/saved_data/temp_tensors'

class Encoder(nn.Module):
    def __init__(self, max_vocab_length: int, num_heads = 3, sequence_length: int = 4, embedding_dim: int = 100, dropout: float = 0.1, hide_target_rate: float = 0.5  ,**kwargs):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(max_vocab_length, embedding_dim)        
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
        self.eval()
    
    def train_mode(self):
        self.hide_target_rate = self.save_hide_target_rate
        self.train()
    
    
    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        r"""[Encode words into vectors]

        Args:
            x (torch.Tensor): [sequence of words, shape: [num_sequences, max_vocab_length]]
            x0 (torch.Tensor): [context words, shape: [num_sequences, sequence_length, max_vocab_length]]

        Returns:
            torch.Tensor: [sequence of vectors, shape: [num_sequences, embedding_dim]]
        """
        #dim reduction
        x01 = self.embedding(x0).squeeze()
        
        #hide target or not
        hide = torch.rand(1)[0]
        if (hide < self.hide_target_rate):
            x1 = torch.zeros(x.shape[0], x01.shape[-1]).cuda()
        else:
            x1 = self.embedding(x).squeeze()
        
        #add positional encoding
        x01 = self.pe(x01)

        #multihead attention
        z1 = self.mha(x01)
        
        #combine all context words into 1 vector
        z1 = self.combine(torch.transpose(z1, 1, 2)).squeeze()

        #add and normalize
        z1 = F.normalize(z1 + x1, dim= 1)
        
        #feadforwad
        z2 = self.fc(z1)
        
        z2 = z2 + z1
        
        return z2
        

class TargetLearner(nn.Module):
    def __init__(self,max_vocab_length:int, embedding_dim:int, **kwargs):
        super(TargetLearner, self).__init__()
        self.embedding = nn.Embedding(max_vocab_length, embedding_dim= embedding_dim)
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
    
        return self.embedding(encoded).squeeze()


class ContextLearner(nn.Module):
    def __init__(self,max_vocab_length:int, embedding_dim:int, sequence_length:int, **kwargs):
        super(ContextLearner, self).__init__()
        self.combine = nn.Sequential(
                        nn.Linear(sequence_length, 1),
                        nn.ReLU(),
                        ) 
        self.embedding = nn.Embedding(max_vocab_length, embedding_dim= embedding_dim)
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        encoded = self.embedding(encoded).squeeze()
        return self.combine(torch.transpose(encoded, 1 , 2)).squeeze()

class WordEmbeddingModel(pl.LightningModule):
    def __init__(self, max_vocab_length:int, embedding_dim: int = 200, num_heads:int = 3, window_size: int = 4, dropout: float= 0.1, lr: float= 1e-4, eps: float= 1e-5, hide_target_rate: float = 0.5, **kwargs):
        super(WordEmbeddingModel, self).__init__()
        self.lr = lr
        self.eps = eps
        self.encode = Encoder(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, sequence_length= 2 * window_size + 1, dropout= dropout, hide_target_rate = hide_target_rate)
        self.context_learner = ContextLearner(embedding_dim= embedding_dim, max_vocab_length= max_vocab_length, sequence_length= 2 * window_size)
        self.target_learner = TargetLearner(embedding_dim= embedding_dim, max_vocab_length= max_vocab_length)
        self.max_vocab_length = max_vocab_length
        self.window_size = window_size
        self.target = torch.Tensor([1]).cuda()

    
    def eval_mode(self):
        self.encode.eval_mode()
        self.eval()
        
    def train_mode(self):
        self.encode.train_mode()
        self.train()
    
    def forward(self, batch):
        x0, x = batch
        out = self.encode(x, x0)
        return out
       
    
    def training_step(self, batch, batch_idx):
        contexts, targets = batch
        out = self.encode(targets, contexts)
        embedded_context = self.context_learner(torch.cat((contexts[:, : self.window_size, : ], contexts[: , self.window_size + 1 : , :]), dim = 1))
        embedded_target = self.target_learner(targets)
        target= torch.Tensor([-1 if x.item() == 0 else 1 for x in targets]).cuda()
        loss1 = F.cosine_embedding_loss(out, embedded_context, self.target)
        loss2 = F.cosine_embedding_loss(out, embedded_target, target)
        loss =1 - (2 * (1 - loss1) * (1 - loss2) / (2 - loss1 - loss2))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print('Epochs {}: loss: {}'.format(self.current_epoch, avg_loss))
        
    @property
    def num_params(self):
        encode_params = sum(p.numel() for p in self.encode.parameters() if p.requires_grad) 
        context_params = sum(p.numel() for p in self.context_learner.parameters() if p.requires_grad)
        target_params = sum(p.numel() for p in self.target_learner.parameters() if p.requires_grad)
        total_params = encode_params + context_params + target_params 
        return encode_params, context_params, target_params, total_params
    
    def configure_optimizers(self):
        
        optimizer_grouped_parameters = [
            {
                "params": p
                    for p in self.encode.parameters()
            },
            {
                "params": p
                    for p in self.context_learner.parameters()
            },
            {
                "params": p
                    for p in self.target_learner.parameters()
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
        load_embedder: bool = True,
        vocab_builder: VocabBuilder = None, 
        max_vocab_length: int = 20000, 
        embedding_dim: int = 200, 
        num_heads: int = 3, 
        dropout: float = 0.1, 
        lr: float = 1e-4,
        eps: float = 1e-5, 
        window_size: int = 3,
        gpus: int = 1,
        hide_target_rate: float = 0.5,
        cfg_optimizer: bool= False,
        use_lr_finder:bool = True,
        ):
        self.use_lr_finder= use_lr_finder

        if load_embedder:
            if cfg_optimizer:
                self.load(lr= lr, eps= eps, hide_target_rate= hide_target_rate, dropout= dropout)
            else:   
                self.load()
            self.vocab_builder = vocab_builder
            self.gpus = gpus
            self.hprams = None
            
        else:
            self.model = WordEmbeddingModel(max_vocab_length= max_vocab_length, embedding_dim= embedding_dim, num_heads= num_heads, window_size= window_size, dropout= dropout, lr = lr, eps = eps, hide_target_rate= hide_target_rate)
            self.hprams = locals()
            del self.hprams['self']
            del self.hprams['vocab_builder']
            del self.hprams['load_embedder']
            del self.hprams['cfg_optimizer']
            del self.hprams['use_lr_finder']
            self.window_size = window_size
            self.vocab_builder = vocab_builder
            self.max_vocab_length = max_vocab_length
            self.gpus = gpus
            if self.gpus == 0:
                print('This model requires at least 1 gpu!')
                del self
                return
        print(self)
        
        
    def __str__(self) -> str:
        encode_params, context_params, target_params, total = self.model.num_params
        info = 'Weights summary\n==========================================\n'
        info += f'Encoder: {(encode_params / 1e6):.1f} M\n'
        info += f'Context Leaner: {(context_params / 1e6):.1f} M\n'
        info += f'Target Learner: {(target_params / 1e6):.1f} M\n'
        info += '==========================================\n'
        info += f'Total: {(total /1e6):.1f} M\n'
        return info
    
        
    def setup_trainer(self, gpus, epochs):
        self.trainer = Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 5)
    
    def setup_data(self, split_index: int, texts: list, labels:list= None, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, inference = False, dataset_splits: int = 10):
        self.count +=1
        if inference:
            dataset = InferenceDataset(labels= labels, split_index= split_index, dataset_splits = dataset_splits, texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
            self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False, pin_memory= pin_memory, num_workers= num_workers)
            self.text_ends = dataset.get_text_ends()
            self.labels = dataset.get_labels()
        else:
            dataset = EmbedDataset(split_index= split_index, dataset_splits= dataset_splits, texts = texts, vocab_builder= self.vocab_builder, max_vocab_length= self.max_vocab_length, window_size= self.window_size)
            self.data_loader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True, pin_memory= pin_memory, num_workers= num_workers)
        
        if self.count == dataset_splits:
            del dataset
    
    def fit(self, texts: list, epochs: int = 20, batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, 
            dataset_splits: int = 10):
        self.count = 0
        #turn on train mode
        self.setup_trainer(gpus= self.gpus, epochs = epochs)
        self.setup_data(texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, dataset_splits= dataset_splits, split_index= self.count)
        if self.use_lr_finder:
            lr_finder= self.trainer.tuner.lr_find(self.model, train_dataloaders= self.data_loader)
            self.model.lr = lr_finder.suggestion()
            print(f'Learning rate= {self.model.lr}')
        for i in range(dataset_splits):
            s = time.time()
            #prepare data
            if i != 0:
                self.setup_trainer(gpus= self.gpus, epochs = epochs)
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
    
    def load_vocab_builder(self, vocab_builder: VocabBuilder):
        self.vocab_builder = vocab_builder
    
    def save(self):
        #save network
        self.trainer.save_checkpoint(dir_path + model_file, weights_only= True)
        if self.hprams:
            #save model hprams
            with open(dir_path + hprams_file, 'wb+') as f:
                pickle.dump(self.hprams, f)
        print('Saved word embedder')
        
    def load(self, **optim_params):
        print('Loading word embedder...')
        kwargs = None
        with open (dir_path + hprams_file, 'rb') as f:
            kwargs = pickle.load(f)
        if kwargs:
            if optim_params:
                for attribute in optim_params.keys():
                    kwargs[attribute] = optim_params[attribute]
            max_vocab_length = kwargs['max_vocab_length']
            lr = kwargs['lr']
            eps = kwargs['eps']
            window_size = kwargs['window_size']
            hiden_target_rate = kwargs['hide_target_rate']
            num_heads = kwargs['num_heads']
            dropout = kwargs['dropout']
            embedding_dim = kwargs['embedding_dim']
            self.max_vocab_length = max_vocab_length
            self.window_size = window_size
            self.model =  WordEmbeddingModel.load_from_checkpoint(dir_path + model_file, max_vocab_length= max_vocab_length, lr= lr, eps= eps, window_size = window_size, hiden_target_rate= hiden_target_rate, num_heads= num_heads, dropout= dropout, embedding_dim= embedding_dim)
        else:
            print('No embedder found')
        
    def embed(self, texts: list, labels: list, batch_size: int = 512, num_workers: int = 4, pin_memory: bool = True, dataset_splits: int = 1, is_train_set: bool= True, index_start: int = 1):
        r"""[embed input texts]

        Args:
            texts (list): [list of raw texts]

        Returns:
            embedded tensors, labels saved in files
        """
        index_start -= 1
        self.count = index_start
        self.setup_trainer(gpus= self.gpus, epochs= 1)
        self.model.cuda()
        self.model.eval_mode()
        self.flag = False
        info = {}
        #remove existed tensors
        if is_train_set:
            name = '/train_'
            spare = 'test'
        else:
            name = '/test_'
            spare = 'train'
            
        if index_start == 0:
            try:
                if is_train_set:
                    os.remove(dir_path + train_info_file)
                else:
                    os.remove(dir_path + test_info_file)
            except:
                pass
            remove_file_in_folders(dir_path + tensors_folder, spare= spare)
        for i in range(index_start, dataset_splits):
            #prepare data
            self.setup_data(labels= labels, texts= texts, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory, inference= True, split_index= self.count, dataset_splits= dataset_splits)
            #embed
            words = torch.cat(self.trainer.predict(self.model, self.data_loader, return_predictions= True)).cpu()
            info[i + 1] = {'text_ends': self.text_ends, 'labels': self.labels}
            #save tensors
            print(f'Saving dataset {i + 1} ...')
            with open(dir_path + tensors_folder + name +'tensor_dataset_' + str(i + 1) + 'outof' + str(dataset_splits), 'wb+') as f:
                torch.save(words, f)
                del words
            if is_train_set:
                try:
                    if index_start != 0:
                        with open(dir_path + train_info_file, 'rb') as f:
                            try:
                                prev = pickle.load(f)
                                for index in prev.keys():
                                    info[index] = prev[index]
                            except:
                                pass
                except:
                    pass
                with open(dir_path + train_info_file, 'wb+') as f:
                    pickle.dump(info, f)
            else:
                try:
                    if index_start != 0:
                        with open(dir_path + test_info_file, 'rb') as f:
                            try:
                                prev = pickle.load(f)
                                for index in prev.keys():
                                    info[index] = prev[index]
                            except:
                                pass
                except:
                    pass
                with open(dir_path + test_info_file, 'wb+') as f:
                    pickle.dump(info, f)
            print(f'Dataset{i + 1} saved')
                
            #if gpu run out of memory, decrease batch size by a half
            if self.flag:
                batch_size = batch_size / 2
                self.flag = False
        
        
        
        
        print('Data saved')
        
        
    
