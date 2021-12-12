
import torch
from torch.utils.data import Dataset
from .VocabularyBuilder import VocabBuilder
import string
from tqdm import tqdm

labels_dict = {
        'Sức khỏe': 0,
        'Ô tô xe máy': 1,
        'Giải trí': 2,
        'Giáo dục': 3,
        'Pháp luật': 4,
        'Số hóa': 5,
        'Đời sống': 6,
        'Du lịch': 7,
        'Thể thao': 8,      
        'Khoa học': 9,
        'Kinh doanh': 10,
        'Thế giới': 11,
        'Thời sự': 12,
}
  
    
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
                    if w not in string.punctuation:
                        words.append(w)
            for i in range (window_size, len(words) - window_size):
                end = min(i + window_size + 1, len(words))
                self.contexts.append(self.transform(words[i - window_size : end]))
                self.targets.append(self.vocab_builder.one_hot(words[i], self.max_vocab_length, return_one_hot= False))

        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        if self.contexts.shape[0] < window_size * 2 + 1:
            self.contexts = torch.cat(self.contexts, torch.zeros(window_size * 2 + 1 - self.contexts.shape[0], self.contexts.shape[1], self.contexts.shape[2]))
        self.targets = torch.stack(self.targets)                

        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        one_hots = [self.vocab_builder.one_hot(word, self.max_vocab_length, return_one_hot= False) for word in words]
        one_hots = torch.stack(one_hots)
        return one_hots
    
                
class InferenceDataset(Dataset):
    def __init__(self,texts: list, labels: list, vocab_builder: VocabBuilder, max_vocab_length: int = 20000, window_size: int = 5, dataset_splits:int = 10, split_index: int = 0):
        r"""[prepare data for embedding before training classifier]

        Args:
            texts (list): [list of raw texts]
            vocab_builder (VocabBuilder): [vocab builder]
            max_vocab_length (int, optional): [the number of words of vocab use for embedding]. Defaults to 20000.
            window_size (int, optional): [the number of context words each side]. Defaults to 5.
        """
        super(InferenceDataset, self,).__init__()
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
        labels = labels[start : end]
        self.text_ends = [0]
        self.labels = []
        text_end = 0
        for text in tqdm(texts):
            self.labels.append(labels_dict[labels[len(self.text_ends) - 1]])
            #tokenize
            wordz = self.vocab_builder.tokenize(text)
            words = []
            for word in wordz:
                for w in word:
                    if w not in string.punctuation:
                        words.append(w)
            s = len(self.targets)
            for i in range (window_size, len(words) - window_size):
                end = min(i + window_size + 1, len(words))
                self.contexts.append(self.transform(words[i - window_size : end]))
                self.targets.append(self.vocab_builder.one_hot(words[i], self.max_vocab_length, return_one_hot= False))
            t = len(self.targets)
            text_end += t - s
            self.text_ends.append(text_end)

        #transform into tensors
        self.contexts = torch.stack(self.contexts)
        if self.contexts.shape[0] < window_size * 2 + 1:
            self.contexts = torch.cat(self.contexts, torch.zeros(window_size * 2 + 1 - self.contexts.shape[0], self.contexts.shape[1], self.contexts.shape[2]))
        self.targets = torch.stack(self.targets)    
        
    def __len__(self):
        return self.contexts.shape[0]
    
    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
    
    def transform(self, words: list):       
        #transform into BOW form
        one_hots = [self.vocab_builder.one_hot(word, self.max_vocab_length, return_one_hot= False) for word in words]
        one_hots = torch.stack(one_hots)
        return one_hots
    
    def get_text_ends(self):
        return self.text_ends
    
    def get_labels(self):
        return self.labels

class ClassifierInputDataset(Dataset):
    def __init__(self, input_tensor: torch.Tensor, text_ends: list, labels: torch.Tensor = None):
        self.labels = labels
        super(ClassifierInputDataset, self).__init__()
        self.texts = []
        for i in range(1, len(text_ends)):
            self.texts.append(torch.clone(input_tensor[text_ends[i - 1] : text_ends[i]]).detach())        
    
    def __len__(self):
        return len(self.texts)
    
    @property
    def embedding_dim(self):
        return self.texts[0].shape[-1]
    
    def __getitem__(self, index):
        if self.labels:
            return self.texts[index], self.labels[index]
        else:
            return self.texts[index]