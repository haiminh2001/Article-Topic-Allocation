
from numpy.core.numeric import indices
import torch
import pytorch_lightning as pl
import numpy as np
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
info_test_file =  '/data_module/saved_data/embed_test_info.pickle'
num_labels = 13
hprams_file= '/data_module/saved_data/classifer_hprams.pickle'
model_file= '/data_module/saved_data/classifier.ckpt'

def text_permute(x: torch.Tensor, permute_amount: float, text_length: int, sequence_length: int):
    total_sequences = int ( text_length / sequence_length)
    num_sequences = int (total_sequences * permute_amount)
    padding = text_length - total_sequences * sequence_length
    if padding != 0:
        padding = sequence_length - padding
        x = torch.cat((x, torch.zeros(x.shape[0], padding, x.shape[2]).cuda()), dim= 1)
        total_sequences +=1
        
    x = x.reshape(x.shape[0], total_sequences, sequence_length, x.shape[2])
    indices1 = np.random.choice(total_sequences, num_sequences ,replace= False)
    indices0 = np.sort(indices1)
    x[:,indices0,:,:] = x[:,indices1,:,:]
    x = x.reshape(x.shape[0], total_sequences * sequence_length, x.shape[-1])
    if padding != 0:
        x = x[:, : text_length, :]
    return x

def text_rotation(x: torch.Tensor):
    text_length = x.shape[1]
    pivot = torch.randint(low = 0, high =text_length, size = (1, )).item()
    x = torch.cat((x[:, pivot : , :], x[: , : pivot, :]), dim = 1)
    return x
  

class Classifier():
    def __init__(self,
                 num_workers:int = 4, 
                 dropout:float = 0.1,
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
            self.dropout = dropout
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
        feature_map = torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index + 1}outof{total}')
        dataset= ClassifierInputDataset(input_tensor= feature_map, text_ends= info['text_ends'], labels= info['labels'])
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
        del feature_map
        del dataset
        del train_dataset
        del valid_dataset
    
    def setup_test_data(self, index: int= 0):    
        total = self.num_test_datasets 
        print(f'Dataset: {index + 1}/{total}')
        #get text ends
     
        with open(dir_path + info_test_file, 'rb') as f:
            info = pickle.load(f)[index + 1]
        
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/test_tensor_dataset_{index + 1}outof{total}'), text_ends= info['text_ends'], labels= info['labels'])
        self.embedding_dim = dataset.embedding_dim
        try:
            self.hprams['embedding_dim'] = self.embedding_dim
            self.hprams['output_dim']= num_labels
        except:
            pass
        #get tensor, labels and create dataset
        test_data_loader= DataLoader(
            dataset= dataset,
            num_workers= self.num_workers,
            batch_size= self.eval_batch_size,
        ) 
        del dataset
        return test_data_loader
              
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
            
    
    def setup_trainer(self, gpus, epochs= 0):
        if epochs ==0:
            self.trainer = pl.Trainer(gpus = gpus, weights_summary=None, log_every_n_steps= 1, num_sanity_val_steps=0)
        else:
            self.trainer = pl.Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 1, num_sanity_val_steps=0)
    
    def fit(self,
            epochs:int= 5,
            train_batch_size: int= 256,
            eval_batch_size: int = 256,
            valid_split:float = 0.2,
            datasets:list = None,
            permute_amount: float = 0.3,
            sequence_length: int = 8,
            permute_rate: float = 0.3,
            rotation_rate: float = 0.1,
            ):
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.valid_split = valid_split
        if datasets:
            self.setup_train_data(self.valid_split, index = datasets[0] - 1)
        else:
            self.setup_train_data(self.valid_split, index= 0)
        self.setup_trainer(self.gpus, self.epochs)
        if self.model_set_upped == False:
                    self.setup_model()
                    self.model_set_upped = True
                    print(self)
                    
        self.classifier.noise_config(permute_amount= permute_amount, sequence_length= sequence_length, permute_rate= permute_rate, rotation_rate = rotation_rate)
        self.classifier.train()
        if self.use_lr_finder:
            
            lr_finder= self.trainer.tuner.lr_find(self.classifier, train_dataloaders= self.train_data_loader)
            self.classifier.lr = lr_finder.suggestion()
            print(f'Learning rate= {self.classifier.lr}')
        
        inputs = datasets if datasets else range(self.num_train_datasets)
        for i in inputs:
            update_data = False
            if datasets:
                if i!= inputs[0]:
                    update_data = True
            else:
                if i!= 0:
                    update_data = True
            if update_data:
                self.setup_train_data(valid_split= self.valid_split, index= i + 1)
                self.setup_trainer(gpus= self.gpus, epochs= self.epochs)
        
            self.trainer.fit(
                model= self.classifier,
                train_dataloaders= self.train_data_loader,
                val_dataloaders= self.valid_data_loader,
            )
            del self.train_data_loader
            del self.valid_data_loader
        
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
    
    
    def test(self, batch_size: int = 256):
        self.eval_batch_size= batch_size
        print('Preparing data:')
        self.setup_trainer(self.gpus)
        if self.model_set_upped == False:
                    self.setup_model()
                    self.model_set_upped = True
                    print(self)
        self.classifier.eval()
        dataloaders = []
        for i in range(self.num_test_datasets):
            dataloaders.append(self.setup_test_data(index= i))
        
        self.trainer.test(
                model= self.classifier,
                dataloaders= dataloaders,
            )
        
    def __str__(self):
        if self.model_set_upped:
            cnn, lstm, fc, total = self.classifier.num_params
            info = 'Weights summary\n==========================================\n'
            info += f'CNN Blocks: {(cnn/1e6):.1f} M\n'
            info += f'Lstm Layers: {(lstm /1e6):.1f} M\n'
            info += f'Fully connected: {(fc /1e3):.1f} K\n'
            info += '==========================================\n'
            info += f'Total: {(total /1e6):.1f} M\n'
            return info
        else:
            return 'Model has not initialized yet!'
       
        
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
    def __init__(self, embedding_dim:int, output_dim:int, lr:float, eps:float):
        super().__init__()
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
        
        self.lstm1 = nn.LSTM(input_size=buffer2, hidden_size = 768, batch_first=True, dropout= 0.1, num_layers= 3)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, output_dim),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.eps = eps
        self.permute_amount: float
        self.sequnece_length: int
        self.permute_rate: float
    
    @property
    def num_params(self):
        cnn_params = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad) 
        lstm_params = sum(p.numel() for p in self.lstm1.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in self.fc.parameters() if p.requires_grad) 
        total_params = cnn_params + lstm_params + fc_params
        return cnn_params, lstm_params, fc_params, total_params
    
    def forward(self, x):
        inp = torch.moveaxis(x, 1, 2)
        inp = self.cnn(inp)
        inp = torch.moveaxis(inp, 1, 2)
        _, (_,inp) = self.lstm1(inp)
        inp = torch.mean(inp, dim= 0).squeeze()
        inp = self.fc(inp)
        return inp
    
    def noise_config(self, permute_amount:float, sequence_length: int, permute_rate: float, rotation_rate: float = 0.3):
        self.permute_amount = permute_amount
        self.sequnece_length = sequence_length 
        self.permute_rate = permute_rate
        self.rotation_rate = rotation_rate
    
    def training_step(self, batch, batchidx):
        texts, labels = batch
        onehot = one_hot(labels, num_labels).type(torch.float)
        if self.permute_amount and self.permute_amount != 0:
            if torch.rand(1).item() < self.permute_rate:
                texts = text_permute(texts, self.permute_amount, text_length= texts.shape[1], sequence_length= self.sequnece_length)
        if self.rotation_rate !=0:
            if torch.rand(1).item() < self.rotation_rate:
                texts = text_rotation(texts)
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
        del pred, labels, avg_loss, avg_acc

    def test_step(self, batch, *_):
        texts, labels = batch
        onehot = one_hot(labels, num_labels).type(torch.float)
        pred = self(texts)
        loss = cross_entropy(pred, onehot)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(pred, dim=1)        
        return {'loss': loss, 'pred': pred, 'labels' : labels}
    
    def test_epoch_end(self, outputs):
        if (type(outputs[0]) != list):
            outputs = [outputs]
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
        

        
        