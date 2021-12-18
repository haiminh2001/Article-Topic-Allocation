
import torch
from torch.utils.data import Dataset
from .VocabularyBuilder import VocabBuilder
from tqdm import tqdm

    
class EmbedDataset(Dataset):
    def __init__(self,dataset_splits: int, split_index: int, texts: list, vocab_builder: VocabBuilder, max_vocab_length: int = 20000, window_size: int = 5):
        r"""
        [Prepare data for training word embedder]
        """
        super(EmbedDataset, self).__init__()
        print('One hot encoding...')
        print(f'Dataset: {split_index + 1}/{dataset_splits}')
        self.vocab_builder = vocab_builder
        self.max_vocab_length = max_vocab_length
        self.contexts = []
        self.targets = []
        start = int(len(texts) / dataset_splits) * split_index
        end = start + int(len(texts) / dataset_splits) + 1
        end = len(texts) if len(texts) <  end  else end 
        texts = texts[start : end]
        for text in tqdm(texts):
            #tokenize
            wordz = self.vocab_builder.tokenize(text)
            words = []
            for word in wordz:
                for w in word:
                        words.append(w)
            for i in range (window_size, len(words) - window_size):
                end = min(i + window_size + 1, len(words))
                self.contexts.append(self.transform(words[i - window_size : end]))

        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        if self.contexts.shape[0] < window_size * 2 + 1:
            self.contexts = torch.cat(self.contexts, torch.zeros(window_size * 2 + 1 - self.contexts.shape[0], self.contexts.shape[1], self.contexts.shape[2]))

        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        bow = [torch.Tensor([self.vocab_builder.get(word, self.max_vocab_length)]).type(torch.long) for word in words]
        bow = torch.stack(bow)
        return bow
    