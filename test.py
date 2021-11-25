import pickle
import os
from VocabularyBuilder import VocabularyBuilder as VB
path = os.path.dirname(os.path.abspath(__file__))
vb = VB(learn = True)
filename =   '/vocabulary.pickle'
vb.fit(['Bành dạo này hơi béo'])

with open(path + filename, 'rb') as f:
    new_dict = pickle.load(f)

print(new_dict)
