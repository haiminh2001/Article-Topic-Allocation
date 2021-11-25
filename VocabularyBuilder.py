import string
from vncorenlp import VnCoreNLP
import os
import pickle


path = os.path.dirname(os.path.abspath(__file__))


class VocabularyBuilder:
  def __init__(self, learn: bool= False, vocab_file: str = '/vocabulary.pickle', **kwargs):
    #learn new vocab or not
    self.learn = learn
    self.vocab_file = vocab_file
    if learn:
        self.annotator: VnCoreNLP = VnCoreNLP(path + "/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')
    
    #load learnt vocab
    with open(path + self.vocab_file, 'rb') as f:
        try:
            self.vocab: dict =  pickle.load(f)
        except:
            self.vocab: dict = {}
  
  def fit(self, texts: list):
    assert self.learn
    for text in texts:
        #remove punctuation
        words = self.annotator.tokenize(text.lower())[0]
        for word in words:
            if word in string.punctuation:
                continue
            if word not in self.vocab.keys():
                self.vocab[word] = len(self.vocab)
    
    with open(path + self.vocab_file, 'wb+') as f:
        pickle.dump(self.vocab, f)
        

        
        
        

