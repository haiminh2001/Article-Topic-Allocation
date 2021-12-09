import torch
import pytorch_lightning as pl
from data_module import ClassifierInputDataset, DataHolder
from os.path import dirname, abspath
import pickle
import os
from torch.utils.data import DataLoader
import pandas as pd

dir_path = dirname(dirname(abspath(__file__)))
tensors_folder = '/data_module/saved_data/temp_tensors'
train_ends_file = '/data_module/saved_data/embed_train_ends.pickle'
test_ends_file = '/data_module/saved_data/embed_test_ends.pickle'
class Classifer(pl.LightningModule):
    def __init__(self, data_holder:DataHolder, num_workers:int = 4, train_batch_size:int = 256, eval_batch_size:int = 1024):
        super(Classifer, self).__init__()
        self.num_train_datasets = 0
        self.num_test_datasets = 0
        self.count_dataset()
        self.data_holder = data_holder
        self.train_labels: list
        self.test_labels: list
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.load_labels()
    
    def setup_train_data(self, valid_split:float= 0.2, index:int= 0):
        
        total = self.num_train_datasets 
        print(f'Dataset{index + 1}/{total}')
        #get text ends
     
        with open(dir_path + train_ends_file, 'rb') as f:
            text_ends = pickle.load(f)[index]
  
        #get tensor, labels and create dataset
        dataset= ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index}outof{total}.pt'), text_ends= text_ends, labels= self.train_labels[index])
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [1 - valid_split, valid_split])
        self.train_data_loader = DataLoader(train_dataset, num_workers= self.num_workers, shuffle= True, batch_size= self.train_batch_size)
        self.valid_data_loader = DataLoader(valid_dataset, num_workers= self.num_workers, batch_size= self.eval_batch_size)
        del dataset
        del train_dataset
        del valid_dataset
    
    def setup_test_data(self, index: int= 0):    
        total = self.num_test_datasets 
        print(f'Dataset{index + 1}/{total}')
        #get text ends
     
        with open(dir_path + train_ends_file, 'rb') as f:
            text_ends = pickle.load(f)[index]
  
        #get tensor, labels and create dataset
        self.test_data_loader= DataLoader(
            ClassifierInputDataset(input_tensor= torch.load(dir_path + tensors_folder + f'/train_tensor_dataset_{index}outof{total}.pt'), text_ends= text_ends, labels= self.test_labels[index]),
            num_workers= self.num_workers,
            batch_size= self.eval_batch_size,
        ) 
        
    def load_labels(self):
        train_labels = self.data_holder.train_labels
        num_train_texts = len(train_labels)
        test_labels = self.data_holder.test_labels
        num_test_texts = len(test_labels)
        self.train_labels = []
        self.test_labels = []
        
        for i in range(self.num_train_datasets):
            start= int(num_train_texts / self.num_train_datasets) * i
            end = start + int(num_train_texts / self.num_train_datasets) + 1
            end = num_train_texts if num_train_texts < end else end
            self.train_labels.append(self.train_labels + pd.DataFrame(train_labels.iloc[start:end]).tolist())
        del train_labels
        
        for i in range(self.num_test_datasets):
            start= int(num_test_texts / self.num_test_datasets) * i
            end = start + int(num_test_texts / self.num_test_datasets) + 1
            end = num_test_texts if num_test_texts < end else end
            self.test_labels.append(self.test_labels + pd.DataFrame(test_labels.iloc[start:end]).tolist())
        del test_labels
            
            
    def count_dataset(self):
        for file in os.listdir(self.data_folder):
            if 'train_tensor' in file:
                self.num_train_datasets+= 1
            if 'test_tensor' in file:
                self.num_test_datasets+= 1