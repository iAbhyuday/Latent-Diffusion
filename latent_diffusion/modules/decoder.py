import torch as pt
from torch import nn
from .resblock import ResBlock

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Decoder, self).__init__()

    
    def forward(self, x): 
        return self.layers(x)