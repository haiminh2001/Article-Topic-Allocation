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
    with open(path + self.vocab_file) as input:
        try:
            self.vocab: dict =  pickle.load(input)
        except:
            self.vocab: dict = {}
  
  def fit(self, texts: list):
    assert self.learn
    for text in texts:
        #remove punctuation
        s = text.translate(str.maketrans('','', string.punctuation))
        words = self.annotator.tokenize(s)[0]
        for word in words:
            if word not in self.vocab.keys():
                self.vocab[word] = len(self.vocab)
    
    with open(path + self.vocab_file, 'wb+') as f:
        pickle.dump(self.vocab, f)
        

        
        
        

