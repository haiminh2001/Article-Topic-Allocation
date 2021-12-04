from abc import ABC
import pandas as pd
import os
path = os.path.dirname(os.path.abspath(__file__))
class DataHolder(ABC):
    def __init__(self, data_folder: str = '/articles_csv'):
        self.data_folder = path + data_folder
        self.read_csv()
    
    def read_csv(self):
        self.data = {'texts': [], 'labels': []}
        self.data = pd.DataFrame(self.data)
        for file in os.listdir(self.data_folder):
            if file.endswith('.csv'):
                df = pd.read_csv(self.data_folder + '/' +  file)
                texts = df.iloc[: , 1 ]
                labels = df.iloc[: , 0 ]
                self.data = self.data.append(pd.DataFrame({'texts': texts, 'labels': labels}))
                
    @property
    def texts(self):
        return self.data['texts']
    
    def __str__(self) -> str:
        info = 'columns: '
        
        for column in self.data.columns:
            info += str(column) + ', '
        
        info += 'rows: ' + str(self.data.shape[0])
        info += '\nnum labels: \n'
        info += str(self.data['labels'].value_counts())
        return info
