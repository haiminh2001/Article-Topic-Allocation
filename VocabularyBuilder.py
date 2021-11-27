import string
from vncorenlp import VnCoreNLP
import os
import pickle
from tqdm import tqdm
import torch
from torch.nn import functional as F


path = os.path.dirname(os.path.abspath(__file__))


class VocabularyBuilder:
    def __init__(self, learn: bool= False, vocab_file: str = '/vocabulary.pickle', **kwargs):
        #learn new vocab or not
        self.learn = learn
        self.vocab_file = vocab_file
        if learn:
            self.annotator: VnCoreNLP = VnCoreNLP(path + "/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
        
        #load learnt vocab
        print('Loading learnt vocab')
        with open(path + self.vocab_file, 'rb') as f:
            try:
                self.vocab: dict =  pickle.load(f)
            except:
                self.vocab: dict = {}
    
    def fit(self, texts: list):
        assert self.learn
        for text in tqdm(texts, desc='Learning new vocabulary progess'):
            #remove punctuation
            words = self.annotator.tokenize(text.lower())[0]
            for word in words:
                if word in string.punctuation:
                    continue
                if word not in self.vocab.values() or len(self.vocab) == 0:
                    self.vocab[word] = len(self.vocab)
        
        with open(path + self.vocab_file, 'wb+') as f:
            pickle.dump(self.vocab, f)

    def erase(self):
        print('Warning: All of learnt vocabulary will be erased: y/n')
        confirm = ''
        while(confirm not in ['y', 'n']):
            confirm = input()
        if confirm == 'y':
            open(path + self.vocab_file, 'w').close()
            print('Erased!')
            
    def one_hot(self, word: str, dim: int = 20000):
        if word in self.vocab.keys():
            return F.one_hot (self.vocab[word], dim)
        else:
            return torch.zeros(dim)
    
    def tokenize(self, sequence: str):
        return self.annotator.tokenize(sequence)
                
            
        
        
        

