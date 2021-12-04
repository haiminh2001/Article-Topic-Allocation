import string
from vncorenlp import VnCoreNLP
import os
import pickle
from tqdm import tqdm
import torch
from torch.nn import functional as F
path = os.path.dirname(os.path.abspath(__file__))

class VocabBuilder:
    def __init__(self, vocab_file: str = '/Tokenizer/vocabulary.pickle', **kwargs):
        self.vocab_file = vocab_file
        self.annotator: VnCoreNLP = VnCoreNLP(path + "/Tokenizer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
        
        #load learnt vocab
        print('Loading learnt vocab')
        with open(path + self.vocab_file, 'rb') as f:
            try:
                self.vocab: dict =  pickle.load(f)
            except:
                self.vocab: dict = {}
    
    def fit(self, texts: list):
   
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
            
    def one_hot(self, word: str, max_vocab_length: int = 20000):
        #only take max_vocab_length word
        if word in self.vocab.keys():
            i = self.vocab[word]
            if i < max_vocab_length:
                return F.one_hot (i, max_vocab_length)
        
        return torch.zeros(max_vocab_length)
    
    def tokenize(self, sequence: str):
        return self.annotator.tokenize(sequence)
                
            
        
        
        

