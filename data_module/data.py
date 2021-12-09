from abc import ABC
import pandas as pd
import os
from tqdm import tqdm
path = os.path.dirname(os.path.abspath(__file__))
class DataHolder(ABC):
    def __init__(self, data_folder: str = '/articles_csv'):
        self.data_folder = path + data_folder
        self.read_csv()
    
    def read_csv(self):
        self.data = {}
        self.data['train'] = {'texts': [], 'labels': []}
        self.data['train'] = pd.DataFrame(self.data)
        self.data['test'] = {'texts': [], 'labels': []}
        print('Loading data...')
        for file in tqdm(os.listdir(self.data_folder)):
            if file.endswith('.csv'):
                df = pd.read_csv(self.data_folder + '/' +  file)
                texts = df.iloc[: , 1 ]
                labels = df.iloc[: , 0 ]
                if file.startswith('test'):
                    self.data['test'] = self.data['test'].append(pd.DataFrame({'texts': texts, 'labels': labels}))
                else:
                    self.data['train'] = self.data['train'].append(pd.DataFrame({'texts': texts, 'labels': labels}))
                
    @property
    def train_texts(self):
        return self.data['train']['texts']
    
    @property
    def train_labels(self):
        return self.data['train']['labels']
    
    @property
    def test_texts(self):
        return self.data['test']['texts']
    
    @property
    def test_labels(self):
        return self.data['test']['texts']
    
    def __str__(self) -> str:
        info = 'Data summary:\ntrain data:\ncolumns: '
        
        for column in self.data['train'].columns:
            info += str(column) + ', '
        
        info += 'rows: ' + str(self.data['train'].shape[0])
        info += '\nnum labels: \n'
        info += str(self.data['train']['labels'].value_counts())
        
        info = '\ntest data:\ncolumns: '
        
        for column in self.data['test'].columns:
            info += str(column) + ', '
        
        info += 'rows: ' + str(self.data['test'].shape[0])
        info += '\nnum labels: \n'
        info += str(self.data['test']['labels'].value_counts())
        return info
