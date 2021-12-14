from vncorenlp import VnCoreNLP
import os
import pickle
import torch
from torch.nn import functional as F
path = os.path.dirname(os.path.abspath(__file__))
from sklearn.feature_extraction.text import CountVectorizer


def tokenize(texts, vocab_builder):
  rs = []
  for text in vocab_builder.tokenize(texts):
      rs = rs + [x for x in text]
      
  return rs
class VocabBuilder:
    def __init__(self, vocab_file: str = '/Tokenizer/vocabulary.pickle', min_df:float= 0.05, max_df:float= 0.8, max_features:int = 40000 ,  **kwargs):
        self.vocab_file = vocab_file
        print('Loading learnt vocab...')
        self.annotator: VnCoreNLP = VnCoreNLP(path + "/Tokenizer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
        tokenizer = lambda x: tokenize(texts= x, vocab_builder= self.annotator)
        #load learnt vocab

        with open(path + self.vocab_file, 'rb') as f:
            try:
                self.vocab: dict =  pickle.load(f)
            except:
                self.vocab: dict = {}
        print(f'Have learnt {len(self.vocab)} words')
        if len(self.vocab) != 0:
            self.vectorizer = CountVectorizer(tokenizer= tokenizer, vocabulary= self.vocab, token_pattern= None, min_df= min_df, max_df= max_df, max_features= max_features)
        else:
            self.vectorizer = CountVectorizer(tokenizer= tokenizer, token_pattern= None,  min_df= min_df, max_df= max_df, max_features= max_features)
        
    
    def fit(self, texts: list):
        
        print('Learning new vocab...')
        self.vectorizer.fit(texts)
        self.vocab = self.vectorizer.vocabulary_
        
        with open(path + self.vocab_file, 'wb+') as f:
            pickle.dump(self.vectorizer.vocabulary_, f)

    def erase(self):
        print('Warning: All of learnt vocabulary will be erased: y/n')
        confirm = ''
        while(confirm not in ['y', 'n']):
            confirm = input()
        if confirm == 'y':
            open(path + self.vocab_file, 'w').close()
            print('Erased!')
            
    def one_hot(self, word: str, max_vocab_length: int = 20000, return_one_hot = True):
        #only take max_vocab_length word
        if return_one_hot:
            i = self.vocab.get(word)
            if i != None:
                i = self.vocab[word]
                if i < max_vocab_length:
                    return F.one_hot(torch.tensor([i]).type(torch.long), max_vocab_length).squeeze().type(torch.float)
               
            return torch.zeros(max_vocab_length).type(torch.float)
        else:
            i = self.vocab.get(word)
            if i != None:
                if i < max_vocab_length:
                    return torch.Tensor([i]).type(torch.long)
            return torch.zeros(1).type(torch.long)
    
    def tokenize(self, sequence: str):
        return self.annotator.tokenize(sequence)
    
    
                
            
        
        
        

