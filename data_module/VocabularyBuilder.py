from vncorenlp import VnCoreNLP
import os
import pickle
from tqdm import tqdm
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
            doc_dict = {}
            words = self.annotator.tokenize(text.lower())
            for list in words:
                for word in list:
                    tf = doc_dict.get(word)
                    if tf == None:
                        doc_dict[word]= 1
                    else:
                        doc_dict[word] += 1
            for word in doc_dict.keys():
                tf = self.vocab.get(word)
                if tf == None:
                    self.vocab[word] = {'tf': 1, 'df': 1}
                else:
                    self.vocab[word]['tf'] += doc_dict[word]
                    self.vocab[word]['df'] += 1
        
        occasional_words = []
        for word in self.vocab.keys():
            if self.vocab[word]['df'] < 10:
                occasional_words.append(word)
            else:
                self.vocab[word] = self.vocab[word]['tf']
        
        for word in occasional_words:
            del self.vocab[word]
             
        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: x[1], reverse=True))   
        self.df = take(max_vocab_length, self.vocab.items())
        self.vocab = dict([(x[0], i + 1) for i, x in enumerate(self.df)])
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
    
    def decode(self, texts: list, max_vocab_length:int):
        self.list = take(max_vocab_length, self.vocab.items())
        rs = ''
        for word in texts:
            rs += self.list[word][0] + ' '
            
        return rs