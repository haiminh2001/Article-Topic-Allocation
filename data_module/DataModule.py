from posixpath import split
import torch
from torch.utils.data import Dataset
from .VocabularyBuilder import VocabBuilder
import string
from tqdm import tqdm

class EmbedDataset(Dataset):
    def __init__(self,dataset_splits: int, split_index: int, texts: list, vocab_builder: VocabBuilder, max_vocab_length: int = 20000, window_size: int = 5):
        """
        [Prepare data for training word embedder]
        """
        super(EmbedDataset, self).__init__()
        self.vocab_builder = vocab_builder
        self.max_vocab_length = max_vocab_length
        self.contexts = []
        self.targets = []
        start = int(len(texts) / dataset_splits) * (split_index - 1)
        end = start + int(len(texts) / dataset_splits) + 1
        end = len(texts) if len(texts) <  end  else end 
        texts = texts[start : end]
        print('One hot encoding...')
        print(f'Dataset: {split_index}/{dataset_splits}')
        for text in tqdm(texts):
            #tokenize
            wordz = self.vocab_builder.tokenize(text)
            words = []
            for word in wordz:
                for w in word:
                    if w not in string.punctuation:
                        words.append(w)
            for i in range (window_size, len(words) - window_size):
                end = min(i + window_size, len(words) -1)
                self.contexts.append(self.transform(words[i - window_size : end]))
                self.targets.append(self.vocab_builder.one_hot(words[i], self.max_vocab_length))

        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        self.targets = torch.stack(self.targets)                

        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        one_hots = [self.vocab_builder.one_hot(word, self.max_vocab_length) for word in words]
        one_hots = torch.mean(torch.cat(one_hots), dim= 0)
        return one_hots
    
                
class InferenceDataset(Dataset):
    def __init__(self,texts: list, vocab_builder: VocabBuilder, max_vocab_length: int = 20000, window_size: int = 5, dataset_splits:int = 10, split_index: int = 0):
        """[prepare data for embedding before training classifier]

        Args:
            texts (list): [list of raw texts]
            vocab_builder (VocabBuilder): [vocab builder]
            max_vocab_length (int, optional): [the number of words of vocab use for embedding]. Defaults to 20000.
            window_size (int, optional): [the number of context words each side]. Defaults to 5.
        """
        super(InferenceDataset, self,).__init__()
        self.vocab_builder = vocab_builder
        self.max_vocab_length = max_vocab_length
        self.contexts = []
        self.targets = []
        start = int(len(texts) / dataset_splits) * split_index
        end = start + int(len(texts) / dataset_splits) + 1
        end = len(texts) if len(texts) <  end  else end 
        texts = texts[start : end]
        #the ends of texts
        self.text_ends = [0]
        end = 0
        print('One hot encoding...')
        print(f'Dataset: {split_index}/{dataset_splits}')
        for text in tqdm(texts):
            #tokenize
            words = self.vocab_builder.tokenize(text)
            n = len(words)
            end += n
            self.text_ends.append(n)
            for i in range (window_size, n - window_size):
                end = min(i + window_size, n - 1)
                self.contexts.append(self.transform(words[i - window_size : end]))
                self.targets.append(self.vocab_builder.one_hot(words[i], self.max_vocab_length))
                
        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        self.targets = torch.stack(self.targets)     
        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        one_hots = [self.vocab_builder.one_hot(word, self.max_vocab_length) for word in words]
        one_hots = torch.mean(one_hots)
        return one_hots
    
    def get_text_ends(self):
        return self.text_ends

class ClassifierInputDataset(Dataset):
    def __init__(self, input_tensor: torch.Tensor, text_ends: list, labels: torch.Tensor = None):
        self.labels = labels
        super(ClassifierInputDataset, self).__init__()
        self.texts = []
        for i in range(1, len(text_ends)):
            self.texts.append(torch.clone(input_tensor[text_ends[i - 1] : text_ends[i]]).detach())
            
    def add_labels(self, labels: torch.Tensor):
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        if self.labels:
            return self.texts[index], self.labels[index]
        else:
            return self.texts[index]