import torch
import pytorch_lightning as pl
from data_module import ClassifierInputDataset, DataHolder
from os.path import dirname, abspath
import pickle
import os
from torch.utils.data import DataLoader

dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
info_train_file = 'data_module/saved_data/embed_test_info.pickle'
info_test_file = 'data_module/saved_data/embed_test_info.pickle'

class Classifer(pl.LightningModule):
    def __init__(self, data_holder:DataHolder, num_workers:int = 4, train_batch_size:int = 256, eval_batch_size:int = 1024):
        super(Classifer, self).__init__()
        print('Collecting data information...')
        self.num_train_datasets = 0
        self.num_test_datasets = 0
        self.count_dataset()
        self.data_holder = data_holder
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        
        
        
    
    def setup_train_data(self, valid_split:float= 0.2, index:int= 0):
        
        total = self.num_train_datasets 
        print(f'Dataset{index + 1}/{total}')
        #get text ends
     
        with open(dir_path + info_train_file, 'rb') as f:
            info = pickle.load(f)[index]
  
        #get tensor, labels and create dataset
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index}outof{total}.pt'), text_ends= info['text_ends'], labels= info['labels'])
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [1 - valid_split, valid_split])
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
  
        #get tensor, labels and create dataset
        self.test_data_loader= DataLoader(
            dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/test_tensor_dataset_{index}outof{total}.pt'), text_ends= info['text_ends'], labels= info['labels']),
            num_workers= self.num_workers,
            batch_size= self.eval_batch_size,
        ) 
              
    def count_dataset(self):
        for file in os.listdir(dir_path + tensors_folder):
            if 'train_tensor' in file:
                self.num_train_datasets+= 1
            if 'test_tensor' in file:
                self.num_test_datasets+= 1