from abc import ABC
import pandas as pd
import os
from tqdm import tqdm
from string import digits, punctuation

def remove_puncs_digits(s: str):
    s = s.translate(str.maketrans('', '', punctuation + digits)).lower()
    return s


def standardize(df: pd.DataFrame):
    df['texts'] = df['texts'].apply(remove_puncs_digits)


path = os.path.dirname(os.path.abspath(__file__))
class DataHolder(ABC):
    def __init__(self, data_folder: str = '/articles_csv'):
        self.data_folder = path + data_folder
        self.read_csv()
    
    def read_csv(self):
        self.data = {}
        self.data['train'] = {'texts': [], 'labels': []}
        self.data['train'] = pd.DataFrame(self.data['train'])
        self.data['test'] = {'texts': [], 'labels': []}
        self.data['test'] = pd.DataFrame(self.data['test'])
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
        
        # shuffle the DataFrame rows
        self.data['train'] = self.data['train'].sample(frac = 1)    
        self.data['test'] = self.data['test'].sample(frac = 1)    
        standardize(self.data['train'])
        standardize(self.data['test'])
                
    @property
    def train_texts(self):
        return self.data['train']['texts'].tolist()
    
    @property
    def train_labels(self):
        return self.data['train']['labels'].tolist()
    
    @property
    def test_texts(self):
        return self.data['test']['texts'].tolist()
    
    @property
    def test_labels(self):
        return self.data['test']['labels'].tolist()
    
    def __str__(self) -> str:
        info = 'Data summary:\ntrain data:\ncolumns: '
        
        for column in self.data['train'].columns:
            info += str(column) + ', '
        
        info += 'rows: ' + str(self.data['train'].shape[0])
        info += '\nnum labels: \n'
        info += str(self.data['train']['labels'].value_counts())
        
        info+= '\ntest data:\ncolumns: '
        
        for column in self.data['test'].columns:
            info += str(column) + ', '
        
        info += 'rows: ' + str(self.data['test'].shape[0])
        info += '\nnum labels: \n'
        info += str(self.data['test']['labels'].value_counts())
        return info
