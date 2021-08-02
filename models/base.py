import torch
import torch.nn as nn
from models.utils import *
from abc import abstractmethod


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def reparameterize(self, mu, log_var):
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    
    @abstractmethod
    def forward(self, input):
        pass

