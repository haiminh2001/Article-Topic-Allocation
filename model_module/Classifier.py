
import torch
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from os.path import dirname, abspath
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import one_hot, cross_entropy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import BertModel, BertConfig

from data_module.VocabularyBuilder import VocabBuilder
from .WordEmbedding import WordEmbedder
from tqdm import tqdm
import string
from sklearn.preprocessing import LabelEncoder

dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
info_train_file = '/data_module/saved_data/embed_train_info.pickle'
info_test_file =  '/data_module/saved_data/embed_test_info.pickle'
hprams_file= '/data_module/saved_data/classifer_hprams.pickle'
model_file= '/data_module/saved_data/classifier.ckpt'

class Classifier():
    def __init__(self,
                 num_workers:int = 4, 
                 train_batch_size:int = 256,
                 eval_batch_size:int = 1024,
                 dropout:float = 0.1,
                 valid_split:float = 0.2,
                 lr:float = 1e-3,
                 eps:float = 1e-5,
                 gpus:int =1, 
                 use_lr_finder:bool = True,
                 load_classifier:bool= False,
                 cfg_optimizer:bool= False,
                 classifier_model:str= 'simple',
                 word_embedder: WordEmbedder= None,
                 num_labels: int= 13,
                 max_length: int = 512,
                 vocab_builder: VocabBuilder= None,
                 ):
        super(Classifier, self).__init__()
        self.use_lr_finder= use_lr_finder
        self.classifier_model = classifier_model
        self.le = None
        self.vocab_builder = vocab_builder
        self.max_vocab_length = word_embedder.max_vocab_length
        self.embedding_dim = word_embedder.embedding_dim
        print('Collecting data information...')
        if load_classifier:
            if cfg_optimizer:
                self.load(lr= lr,
                          eps= eps,
                          dropout= dropout,
                          valid_split= valid_split,
                          train_batch_size= train_batch_size,
                          eval_batch_size= eval_batch_size,
                          )
                
            else:
                self.load()
                self.gpus = gpus
            self.hprams = None
            
        else:
            self.embedding = word_embedder.model.encode
            self.window_size = word_embedder.window_size
            self.hprams = locals()
            del self.hprams['load_classifier']
            del self.hprams['cfg_optimizer']
            del self.hprams['self']
            del self.hprams['use_lr_finder']
            del self.hprams['word_embedder']
            del self.hprams['vocab_builder']
            self.max_length = max_length
            self.num_labels = num_labels
            self.hprams['window_size'] = self.window_size
            self.num_workers = num_workers
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.dropout = dropout
            self.valid_split = valid_split
            self.model_set_upped = False
            self.lr = lr
            self.eps = eps
            self.gpus = gpus

    def setup_model(self):
        if self.classifier_model == 'simple':
            self.classifier = SimpleClassifier(embedding_dim = self.embedding_dim, output_dim= self.num_labels, lr= self.lr, eps = self.eps, winddow_size= self.window_size)
        if self.classifier_model == 'bert':
            self.classifier = Bert(self.embedding_dim, self.num_labels, lr= self.lr, eps = self.eps)
        self.classifier.add_module('embedding', self.embedding)
        del self.embedding
        self.model_set_upped= True
        print(self)
            
    
    def setup_trainer(self, gpus, epochs= 0):
        if epochs ==0:
            self.trainer = pl.Trainer(gpus = gpus, weights_summary=None, log_every_n_steps= 1, num_sanity_val_steps=0)
        else:
            self.trainer = pl.Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 1, num_sanity_val_steps=0)
            
            
    def setup_dataloaders(self, input_ids, labels, batch_size:int= 256, shuffle:bool = True, valid_split:float= 0):
        dataset = SimpleDataset(input_ids= input_ids, labels= labels)
        if valid_split == 0:
            return DataLoader(dataset= dataset, batch_size= batch_size, num_workers= self.num_workers, shuffle= shuffle, pin_memory= True)
        else:
            data_length = dataset.__len__()
            valid_length = int(data_length * valid_split)
            train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [data_length - valid_length , valid_length])
            return DataLoader(dataset= train_dataset, batch_size= batch_size, num_workers= self.num_workers, shuffle= True, pin_memory= True), DataLoader(dataset= valid_dataset, batch_size= self.eval_batch_size, num_workers= self.num_workers, shuffle= False, pin_memory= True)
    
    def fit(self, input_ids: torch.Tensor, labels: torch.Tensor, epochs:int= 5, valid_split:float= 0.2):
        self.epochs = epochs
        self.setup_trainer(self.gpus, self.epochs)
        if self.model_set_upped == False:
                    self.setup_model()
                    self.model_set_upped = True
        self.classifier.train_mode()

        train_data_loader, valid_data_loader = self.setup_dataloaders(input_ids= input_ids, labels= labels, batch_size= self.train_batch_size, shuffle= True, valid_split= valid_split)
        if self.use_lr_finder:
            lr_finder= self.trainer.tuner.lr_find(self.classifier, train_dataloaders= train_data_loader)
            self.classifier.lr = lr_finder.suggestion()
            print(f'Learning rate= {self.classifier.lr}')
        

        
        self.trainer.fit(
            model= self.classifier,
            train_dataloaders= train_data_loader,
            val_dataloaders= valid_data_loader,
        )
        
        self.save()
    
    def forward(self, x):
        return self.classifier(x)
    
    def save(self):
        #save network
        self.trainer.save_checkpoint(dir_path + model_file, weights_only= True)

        if self.hprams:
            #save model hprams
            with open(dir_path + hprams_file, 'wb+') as f:
                pickle.dump(self.hprams, f)
        print('Saved classifier')
        
    def load(self, **optim_params):
        print('Loading classifier...')
        kwargs = None
        with open (dir_path + hprams_file, 'rb') as f:
            kwargs = pickle.load(f)
        if kwargs:
            if optim_params:
                for attribute in optim_params.keys():
                    kwargs[attribute] = optim_params[attribute]
            lr = kwargs['lr']
            eps = kwargs['eps']
            dropout = kwargs['dropout']
            embedding_dim = kwargs['embedding_dim']
            output_dim = kwargs['output_dim']
            self.num_workers = kwargs['num_workers']
            self.train_batch_size = kwargs['train_batch_size']
            self.eval_batch_size = kwargs['eval_batch_size']
            self.valid_split = kwargs['valid_split']
            self.gpus = kwargs['gpus']
            self.window_size = kwargs['window_size']
            self.num_labels = kwargs['num_labels']
            self.max_length = kwargs['max_length']
            self.model_set_upped = True

            self.classifier =  SimpleClassifier.load_from_checkpoint(dir_path + model_file, 
                                                          lr= lr, 
                                                          eps= eps, 
                                                          dropout= dropout,
                                                          embedding_dim= embedding_dim,
                                                          output_dim= output_dim,
                                                          num_labels= self.num_labels,
                                                          )
        else:
            print('No classifier found')
    
    
    def test(self, input_ids: torch.Tensor, labels: torch.Tensor):
        self.setup_trainer(self.gpus)
        if self.model_set_upped == False:
                    self.setup_model()
                    self.model_set_upped = True
        self.classifier.eval_mode()
            
        self.trainer.test(
                model= self.classifier,
                dataloaders= self.setup_dataloaders(input_ids= input_ids, labels= labels, batch_size= self.eval_batch_size, shuffle= False),
            )
        
    def __str__(self):
        if self.model_set_upped:
            cnn, lstm, fc, embedding, total = self.classifier.num_params
            info = 'Weights summary\n==========================================\n'
            info += f'Input Embedding: {(embedding/1e6):.1f} M\n'
            info += f'CNN Blocks: {(cnn/1e6):.1f} M\n'
            info += f'Lstm Layers: {(lstm /1e6):.1f} M\n'
            info += f'Fully connected: {(fc /1e3):.1f} K\n'
            info += '==========================================\n'
            info += f'Total: {((total) /1e6):.1f} M\n'
            return info 
        else:
            return 'Model has not initialized yet!'
        
    def tokenize(self, texts: list, labels: list):
        print('Tokenizing...')
        self.text_ends = [0]
        if self.le == None:
            self.le = LabelEncoder()
            int_labels = self.le.fit_transform(labels)
            self.num_labels = len(list(self.le.classes_))
        else:
            try:
                int_labels = self.le.transform(labels)
            except:
                self.le = LabelEncoder()
                int_labels = self.le.fit_transform(labels)
                self.num_labels = len(list(self.le.classes_))
        text_tensors = []
        idx =0 
        for text in tqdm(texts):
            #tokenize
            sequences =[]
            text_removed = 0
            wordz = self.vocab_builder.tokenize(text)
            words = []
            for word in wordz:
                for w in word:     
                    words.append(w)
            end_text = min(self.max_length, len(words))
            for i in range (self.window_size, end_text):
                end = min(i + self.window_size + 1, len(words))
                sequences.append(self.transform(words[i - self.window_size : end]))
                if sequences[-1].shape[0] < self.window_size * 2 + 1:
                    sequences[-1] = torch.cat((sequences[-1], torch.zeros(self.window_size * 2 + 1 - sequences[-1].shape[0], sequences[-1].shape[1])))
                    
            if len(sequences) != 0:
                sequences= torch.cat(sequences)
            else:
                int_labels = np.delete(int_labels, idx - text_removed)
                text_removed+=1
                idx+=1
                continue
            
            if sequences.shape[0] < self.max_length * (2 * self.window_size + 1):
                sequences = torch.cat((sequences, torch.zeros(self.max_length * (2 * self.window_size + 1) - sequences.shape[0], sequences.shape[1])))
            text_tensors.append(sequences[:512 * (self.window_size * 2 + 1)])
            idx+=1
        text_tensors = torch.stack(text_tensors)
    
        return text_tensors.type(torch.long).squeeze(), torch.from_numpy(int_labels).type(torch.long).squeeze()
     
       
       
    def transform(self, words: list):       
        #transform into BOW form
        bow = [torch.Tensor([self.vocab_builder.get(word, self.max_vocab_length)]).type(torch.long) for word in words]
        bow = torch.stack(bow)
        return bow
        
class DenseBlock(nn.Module):
    def __init__(self, input_size):
        super(DenseBlock, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size, input_size, kernel_size= 1, stride= 1),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size, input_size, kernel_size= 1, stride= 1),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size, input_size, kernel_size= 1, stride= 1),
        )
    
    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x + x1)
        x3 = self.cnn3(x + x1 + x2)
        del x1
        del x2
        return x3

class ReductionBlock(nn.Module):
    def __init__(self, input_size):
        super(ReductionBlock, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size, input_size, kernel_size= 1, stride= 1),
        )
        self.cnn2 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size, input_size, kernel_size= 1, stride= 1),
        )
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(input_size,int( input_size / 2), kernel_size= 1, stride= 1),
        )
        self.pooling = nn.AvgPool1d(2)
    
    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x + x1)
        x3 = self.cnn3(x + x1 + x2)
        x3 = self.pooling(x3)
        del x1
        del x2
        return x3
        
class SimpleClassifier(pl.LightningModule):
    def __init__(self, embedding_dim:int, output_dim:int, lr:float, eps:float, winddow_size: int):
        super().__init__()
        self.window_size = winddow_size
        self.num_labels = output_dim
        buffer1 = int(embedding_dim / 2)
        buffer2 = int(buffer1 / 2)
        self.cnn = nn.Sequential(
            DenseBlock(embedding_dim),
            DenseBlock(embedding_dim),
            ReductionBlock(embedding_dim),
            DenseBlock(buffer1),
            DenseBlock(buffer1),
            ReductionBlock(buffer1),
        )
        
        self.lstm1 = nn.LSTM(input_size=buffer2, hidden_size = buffer2, batch_first=True, dropout= 0.1, num_layers= 3)
        self.lstm2 =  nn.LSTM(input_size = buffer2, hidden_size = 100, batch_first = True, dropout= 0.1)
        self.fc = nn.Sequential(
            nn.Linear(100, output_dim),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.eps = eps
    
    def eval_mode(self):
        self.eval()
        self.embedding.eval_mode()
    
    def train_mode(self):
        self.train()
        self.embedding.train_mode()
    
    @property
    def num_params(self):
        cnn_params = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad) 
        lstm_params = sum(p.numel() for p in self.lstm1.parameters() if p.requires_grad) + sum(p.numel() for p in self.lstm2.parameters() if p.requires_grad) 
        fc_params = sum(p.numel() for p in self.fc.parameters() if p.requires_grad) 
        embedding_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad) 
        total_params = cnn_params + lstm_params + fc_params + embedding_params
        return cnn_params, lstm_params, fc_params, embedding_params, total_params
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.flatten(x).reshape(-1 , 2 * self.window_size + 1)
        inp = self.embedding(x).reshape(batch_size, -1, self.embedding_dim)
        inp = torch.transpose(inp, 1, 2)
        inp = self.cnn(inp)
        inp = torch.transpose(inp, 1, 2)
        out1, _ = self.lstm1(inp) 
        _, (hn2, _) = self.lstm2(out1)
        hn2 = hn2.view(-1, 100) 
        out = self.fc(hn2)
        return out
    
    def training_step(self, batch, batchidx):
        texts, labels = batch
        onehot = one_hot(labels, self.num_labels).type(torch.float)
        pred = self(texts)
        
        loss = cross_entropy(pred, onehot)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(pred, dim=1)        
        
        return {'loss': loss, 'pred': pred, 'labels' : labels}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred = torch.cat([x["pred"] for x in outputs]).squeeze().detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).squeeze().detach().cpu().numpy()
        avg_acc = accuracy_score(labels, pred)
        print('Epochs {}: train_loss: {}, accuracy: {}, f1_score: {}'.format(self.current_epoch, avg_loss, avg_acc, f1_score(labels, pred, average='macro')))
        if (self.current_epoch % 5 == 0 and self.current_epoch > 0):
            print(confusion_matrix(labels, pred))

    def test_step(self, batch, batch_idx, dataloader_idx):
        texts, labels = batch
        onehot = one_hot(labels, self.num_labels).type(torch.float)
        pred = self(texts)
        loss = cross_entropy(pred, onehot)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(pred, dim=1)        
        return {'loss': loss, 'pred': pred, 'labels' : labels}
    
    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            avg_loss = torch.stack([x["loss"] for x in output]).mean()
            pred = torch.cat([x["pred"] for x in output]).squeeze().detach().cpu().numpy()
            labels = torch.cat([x["labels"] for x in output]).squeeze().detach().cpu().numpy()
            avg_acc = accuracy_score(labels, pred)
            print('Dataset {}: loss: {}, accuracy: {}, f1_score: {}'.format(i, avg_loss, avg_acc, f1_score(labels, pred, average='macro')))
            print(confusion_matrix(labels, pred))
    
    def configure_optimizers(self):
      
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr= self.lr,
            eps= self.eps,
        )
        return optimizer
    
    def validation_step(self, batch, batchidx):
        return self.training_step(batch, batchidx)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred = torch.cat([x["pred"] for x in outputs]).squeeze().detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).squeeze().detach().cpu().numpy()
        avg_acc = accuracy_score(labels, pred)
        print('Epochs {}: val_loss: {}, accuracy: {}, f1_score: {}'.format(self.current_epoch, avg_loss, avg_acc, f1_score(labels, pred, average='macro')))
        if self.current_epoch % 5 == 0 and self.current_epoch > 0:
            print(confusion_matrix(labels, pred))
        
class Bert(pl.LightningModule):
    def __init__(self, embedding_dim:int, output_dim:int, lr:float, eps:float):
        super(Bert, self).__init__()
        configuration = BertConfig(hidden_size= 350, num_attention_heads= 10, intermediate_size= 1028,)
        self.l1 = BertModel(configuration)
        self.pre_classifier = torch.nn.Linear(350, 350)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(350, output_dim)
        self.lr = lr
        self.eps = eps
    
    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        out = torch.mean(x[:64], dim= 0,keepdim=True)
        for i in range(64, x.shape[0], 64):
            end = min(i + 64, x.shape[0])
            out = torch.cat((out, torch.mean(x[i : end], dim= 0, keepdim= True)))
        
        out = torch.transpose(out, 0, 1)    
        _, out = self.l1(inputs_embeds = out, return_dict= False)
        
        out = self.pre_classifier(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out 
    
    
    def training_step(self, batch, batchidx):
        texts, labels = batch
        onehot = one_hot(labels, self.num_labels).type(torch.float)
        pred = self(texts)
        
        loss = cross_entropy(pred, onehot)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(pred, dim=1)        
        
        return {'loss': loss, 'pred': pred, 'labels' : labels}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred = torch.cat([x["pred"] for x in outputs]).squeeze().detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).squeeze().detach().cpu().numpy()
        avg_acc = accuracy_score(labels, pred)
        print('Epochs {}: train_loss: {}, accuracy: {}, f1_score: {}'.format(self.current_epoch, avg_loss, avg_acc, f1_score(labels, pred, average='macro')))
        if (self.current_epoch % 5 == 0 and self.current_epoch > 0):
            print(confusion_matrix(labels, pred))

        
    def configure_optimizers(self):
      
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr= self.lr,
            eps= self.eps,
        )
        return optimizer
    
    def validation_step(self, batch, batchidx):
        return self.training_step(batch, batchidx)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred = torch.cat([x["pred"] for x in outputs]).squeeze().detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).squeeze().detach().cpu().numpy()
        avg_acc = accuracy_score(labels, pred)
        print('Epochs {}: val_loss: {}, accuracy: {}, f1_score: {}'.format(self.current_epoch, avg_loss, avg_acc, f1_score(labels, pred, average='macro')))
        if self.current_epoch % 5 == 0 and self.current_epoch > 0:
            print(confusion_matrix(labels, pred))
        
class SimpleDataset(Dataset):
    def __init__(self, input_ids, labels):
        super().__init__()
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]
        
        