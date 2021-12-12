import torch
import pytorch_lightning as pl
from torch.nn.modules import dropout
from data_module import ClassifierInputDataset, DataHolder
from os.path import dirname, abspath
import pickle
import os
from torch.utils.data import DataLoader
from torch import nn

dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
info_train_file = 'data_module/saved_data/embed_test_info.pickle'
info_test_file = 'data_module/saved_data/embed_test_info.pickle'
num_labels = 13

class Classifer(pl.LightningModule):
    def __init__(self, data_holder:DataHolder, num_workers:int = 4, train_batch_size:int = 256, eval_batch_size:int = 1024, dropout:float = 0.1, valid_split:float = 0.2, sequence_length: int = 20, sequence_overlapping:int = 3):
        super(Classifer, self).__init__()
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
        self.ie = ImportantEvaluate(embedding_dim= self.embedding_dim, output_dim= num_labels, dropout= self.dropout)
        
    def forward(self, text_tensor):
        pass
        
                
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
        

        
        