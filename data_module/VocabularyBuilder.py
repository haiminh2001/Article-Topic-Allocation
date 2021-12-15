from vncorenlp import VnCoreNLP
import os
import pickle
from tqdm import tqdm
import torch
from torch.nn import functional as F
path = os.path.dirname(os.path.abspath(__file__))

from itertools import islice

def take(n, iterable):
    return list(islice(iterable, n))

class VocabBuilder:
    def __init__(self, vocab_file: str = '/Tokenizer/vocabulary.pickle', **kwargs):
        self.vocab_file = vocab_file
        print('Loading learnt vocab...')
        self.annotator: VnCoreNLP = VnCoreNLP(path + "/Tokenizer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
        
        #load learnt vocab
        with open(path + self.vocab_file, 'rb') as f:
            try:
                self.vocab: dict =  pickle.load(f)
            except:
                self.vocab: dict = {}
        print(f'Have learnt {len(self.vocab)} words')
    
    def fit(self, texts: list, max_vocab_length:int= 40000):
   
        for text in tqdm(texts, desc='Learning new vocabulary progess'):
            #remove punctuation
            words = self.annotator.tokenize(text.lower())
            for list in words:
                for word in list:
                    df = self.vocab.get(word)
                    if df == None:
                        self.vocab[word] = 1
                    else:
                        self.vocab[word] += 1
                        
        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))   
        self.df = take(max_vocab_length, self.vocab.items())
        self.vocab = dict([(x[0], i) for i, x in enumerate(self.df)])
        self.df = dict(self.df)
        
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
            
    def get(self, word:str, max_vocab_length):
        rs = self.vocab.get(word)
        if rs != None and rs < max_vocab_length:
            return rs
        else:
            return 0
    
    def tokenize(self, sequence: str):
        return self.annotator.tokenize(sequence)