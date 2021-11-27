import torch
from torch.utils.data import Dataset
from VocabularyBuilder import VocabularyBuilder

class EmbedDataset(Dataset):
    def __init__(self, texts: list, vocab_builder: VocabularyBuilder, vocab_length: int = 20000, window_size: int = 5):
        """
        [Prepare data for training word embedder]
        """
        super(EmbedDataset, self).__init__()
        self.vocab_builder = vocab_builder
        self.vocab_length = vocab_length
        self.contexts = []
        self.targets = []
        for text in texts:
            #tokenize
            words = self.vocab_builder.tokenize(text)
            
            for i in range (window_size, len(words) - window_size):
                end = min(i + window_size, len(words) -1)
                self.contexts.append(self.transform(words[i - window_size : end]))
                self.targets.append(self.vocab_builder.one_hot(words[i], self.vocab_length))

        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        self.targets = torch.stack(self.targets)                

        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        one_hots = [self.vocab_builder.one_hot(word, self.vocab_length) for word in words]
        one_hots = torch.mean(one_hots)
        return one_hots