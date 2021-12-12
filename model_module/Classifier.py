from pytorch_lightning import trainer
import torch
import pytorch_lightning as pl
from torch.nn.modules import dropout
from data_module import ClassifierInputDataset, DataHolder
from os.path import dirname, abspath
import pickle
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import normalize, one_hot, cross_entropy


dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
info_train_file = 'data_module/saved_data/embed_test_info.pickle'
info_test_file = 'data_module/saved_data/embed_test_info.pickle'
num_labels = 13

class Classifier(pl.LightningModule):
    def __init__(self, data_holder:DataHolder, 
                 num_workers:int = 4, 
                 train_batch_size:int = 256,
                 eval_batch_size:int = 1024,
                 dropout:float = 0.1,
                 valid_split:float = 0.2,
                 sequence_length: int = 20,
                 sequence_overlapping:int = 3,
                 lr:float = 1e-3,
                 eps:float = 1e-5,
                 gpus:int =1, 
                 epochs:int = 5,
                 ):
        super(Classifier, self).__init__()
        print('Collecting data information...')
        self.num_train_datasets = 0
        self.num_test_datasets = 0
        self.count_dataset()
        self.data_holder = data_holder
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.valid_split = valid_split
        self.sequence_length = sequence_length
        self.sequnce_overlapping = sequence_overlapping
        self.model_set_upped = False
        self.lr = lr
        self.eps = eps
        self.gpus = gpus
        self.epochs = epochs
    
    def setup_train_data(self, valid_split:float= 0.2, index:int= 0):
        
        total = self.num_train_datasets 
        print(f'Dataset{index + 1}/{total}')
        #get text ends
     
        with open(dir_path + info_train_file, 'rb') as f:
            info = pickle.load(f)[index]
  
        #get tensor, labels and create dataset
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index}outof{total}.pt'), text_ends= info['text_ends'], labels= info['labels'])
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [1 - valid_split, valid_split])
        self.embedding_dim = dataset.embedding_dim
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
        
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/test_tensor_dataset_{index}outof{total}.pt'), text_ends= info['text_ends'], labels= info['labels']),
        self.embedding_dim = dataset.embedding_dim
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
        self.classifier = SimpleClassifier(self.embedding_dim, self.dropout, num_labels)
        
    def forward(self, text_tensor):
        return self.classifier(text_tensor)
    
    def training_step(self, batch, batchidx):
        texts, labels = batch
        onehot = one_hot(labels, num_labels)
        pred = self.classifier(texts)
        loss = cross_entropy(pred, onehot)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(pred, dim=1)        
        acc = torch.sum(pred == labels) / texts.shape[0]
        
        return {'loss': loss, 'accuracy': acc}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        print('Epochs {}: loss: {}, accuracy: {}'.format(self.current_epoch, avg_loss, avg_acc))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr= self.lr,
            eps= self.eps,
        )
        return optimizer
    
    def validation_step(self, batch, batchidx):
        return self.training_step(batch, batchidx)
    
    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs)
        
    def setup_trainer(self, gpus, epochs):
        self.trainer = pl.Trainer(gpus = gpus, max_epochs= epochs, weights_summary=None, log_every_n_steps= 5)
    
    def fit(self):
        self.setup_trainer(self.gpus, self.epochs)
        if self.model_set_upped == False:
            self.setup_model()
            self.model_set_upped = True
            
        for i in range(self.num_train_datasets):
            self.setup_train_data(self.valid_split, index= i)
            self.trainer.fit(
                model= self.classifier,
                train_dataloaders= self.train_data_loader,
                val_dataloaders= self.valid_data_loader,
            )
            
        
class SimpleClassifier(nn.Module):
    def __init__(self, embedding_dim:int, dropout:float, output_dim:int):
        super().__init__()
        self.importance_eval = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ) 
        self.combine = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Dropout(dropout),
                nn.ReLU,
            ) 

        self.conclude = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Dropout(dropout),
                nn.ReLU,
                nn.Linear(50, output_dim),
                nn.Dropout(dropout),
                nn.ReLU(),  
            ) 
        self.embedding_dim = embedding_dim
    
    def forward(self, text_tensor):
        result = torch.zeros(self.embedding_dim)
        for i in range(text_tensor.shape[0]):
            x = self.importance_eval(text_tensor[i].squeeze())
            result =  normalize(self.combine(torch.cat((result, x))))
        return self.conclude(result)
        
    
        
class ImportantEvaluate(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, dropout:float= 0.1):
        super().__init__()
        self.conv =nn.Sequential(
                nn.Conv1d(in_channels= embedding_dim, out_channels= 64, kernel_size= 3, padding= 1),
                nn.BatchNorm1d(64),
                nn.Conv1d(64,32,3, padding= 1),
                nn.BatchNorm1d(32),
            ) 
        
        self.lstm = nn.LSTM(batch_first= True, input_size = 32, hidden_size = 32)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""[evaluate the importance of a batch of sequences of words in determining the label of the text]

        Args:
            x (torch.Tensor): [shape [num_sequences, sequence_length, embedding_dim]]

        Returns:
            torch.Tensor: [shape [num_sequences, output_dim]]
        """
        out = self.conv(torch.moveaxis(x, 1, 2))
        out = self.lstm(torch.moveaxis(out, 1, 2))
        out = self.fc(out)
        return out
        

        
        