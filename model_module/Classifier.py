
import torch
import pytorch_lightning as pl
from data_module import ClassifierInputDataset
from os.path import dirname, abspath
import pickle
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import one_hot, cross_entropy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import BertModel, BertConfig

dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
info_train_file = '/data_module/saved_data/embed_train_info.pickle'
info_test_file = '/data_module/saved_data/embed_test_info.pickle'
num_labels = 13
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
                 ):
        super(Classifier, self).__init__()
        self.use_lr_finder= use_lr_finder
        self.num_train_datasets = 0
        self.num_test_datasets = 0
        self.count_dataset()
        self.classifier_model = classifier_model
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
            self.hprams = locals()
            del self.hprams['load_classifier']
            del self.hprams['cfg_optimizer']
            del self.hprams['self']
            del self.hprams['use_lr_finder']
            self.num_workers = num_workers
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.dropout = dropout
            self.valid_split = valid_split
            self.model_set_upped = False
            self.lr = lr
            self.eps = eps
            self.gpus = gpus
    
    def setup_train_data(self, valid_split:float= 0.2, index:int= 0):
        
        total = self.num_train_datasets 
        print(f'Dataset{index + 1}/{total}')
        #get text ends
     
        with open(dir_path + info_train_file, 'rb') as f:
            info = pickle.load(f)[index + 1]

        
        #get tensor, labels and create dataset
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index + 1}outof{total}'), text_ends= info['text_ends'], labels= info['labels'])
        data_length = dataset.__len__()
        valid_length = int(data_length * valid_split)
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [data_length - valid_length , valid_length])
        self.embedding_dim = dataset.embedding_dim
        try:
            self.hprams['embedding_dim'] = self.embedding_dim
            self.hprams['output_dim']= num_labels
        except:
            pass
        self.train_data_loader = DataLoader(train_dataset, num_workers= self.num_workers, shuffle= True, batch_size= self.train_batch_size)
        self.valid_data_loader = DataLoader(valid_dataset, num_workers= self.num_workers, batch_size= self.eval_batch_size)
        del dataset
        del train_dataset
        del valid_dataset
    
    def setup_test_data(self, index: int= 0):    
        total = self.num_test_datasets 
        print(f'Dataset: {index + 1}/{total}')
        #get text ends
     
        with open(dir_path + info_test_file, 'rb') as f:
            info = pickle.load(f)[index]
        
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/test_tensor_dataset_{index + 1}outof{total}'), text_ends= info['text_ends'], labels= info['labels']),
        self.embedding_dim = dataset.embedding_dim
        try:
            self.hprams['embedding_dim'] = self.embedding_dim
            self.hprams['output_dim']= num_labels
        except:
            pass
        #get tensor, labels and create dataset
        self.test_data_loader= DataLoader(
            dataset= dataset,
            num_workers= self.num_workers,
            batch_size= self.eval_batch_size,
        ) 
        del dataset
              
    def count_dataset(self):
        for file in os.listdir(dir_path + tensors_folder):
            if 'train_tensor' in file:
                self.num_train_datasets+= 1
            if 'test_tensor' in file:
                self.num_test_datasets+= 1
    
    def setup_model(self):
        if self.classifier_model == 'simple':
            self.classifier = SimpleClassifier(self.embedding_dim, num_labels, lr= self.lr, eps = self.eps)
        if self.classifier_model == 'bert':
            self.classifier = Bert(self.embedding_dim, num_labels, lr= self.lr, eps = self.eps)
            
    
    def setup_trainer(self, gpus, epochs):
        self.trainer = pl.Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 1, num_sanity_val_steps=0)
    
    def fit(self, epochs:int= 5):
        self.epochs = epochs
        self.setup_train_data(self.valid_split, index= 0)
        self.setup_trainer(self.gpus, self.epochs)
        if self.model_set_upped == False:
                    self.setup_model()
                    self.model_set_upped = True
        self.classifier.train()
        if self.use_lr_finder:
            
            lr_finder= self.trainer.tuner.lr_find(self.classifier, train_dataloaders= self.train_data_loader)
            self.classifier.lr = lr_finder.suggestion()
            print(f'Learning rate= {self.classifier.lr}')
            
        for i in range(self.num_train_datasets):
            if i !=0:
                self.setup_train_data(self.valid_split, index= i)
                self.setup_trainer(self.gpus, self.epochs)
        
            self.trainer.fit(
                model= self.classifier,
                train_dataloaders= self.train_data_loader,
                val_dataloaders= self.valid_data_loader,
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
            self.model_set_upped = True

            self.classifier =  SimpleClassifier.load_from_checkpoint(dir_path + model_file, 
                                                          lr= lr, 
                                                          eps= eps, 
                                                          dropout= dropout,
                                                          embedding_dim= embedding_dim,
                                                          output_dim= output_dim,
                                                          )
        else:
            print('No classifier found')
            
        
class SimpleClassifier(pl.LightningModule):
    def __init__(self, embedding_dim:int, output_dim:int, lr:float, eps:float):
        super().__init__()
        input_size = embedding_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm1d(input_size),
            nn.AvgPool1d(2),
            nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm1d(input_size),
            nn.AvgPool1d(2),
        )
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size = 50, batch_first=True) #CNNLSTM
        self.lstm2 =  nn.LSTM(input_size = 50, hidden_size = 50, batch_first = True)
        self.fc1 = nn.Linear(50, output_dim) #fully connected last layer
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.eps = eps
    
    def forward(self, x):
        inp = torch.moveaxis(x, 1, 2)
        inp = self.cnn(inp)
        inp = torch.moveaxis(inp, 1, 2)
        out1, _ = self.lstm1(inp) #CNNLSTM with input, hidden, and internal state
        _, (hn2, _) = self.lstm2(out1)
        hn2 = hn2.view(-1, 50) #reshaping the data for Dense layer next
        out = self.relu(hn2)
        out = self.fc1(out)
        return out
    
    
    def training_step(self, batch, batchidx):
        texts, labels = batch
        onehot = one_hot(labels, num_labels).type(torch.float)
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
        onehot = one_hot(labels, num_labels).type(torch.float)
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
        

        
        