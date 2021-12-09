import torch
import pytorch_lightning as pl

class Classifer(pl.LightningModule):
    def __init__(self):
        super(Classifer, self).__init__()
        