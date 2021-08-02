import torch
import torch.nn as nn

from models import BaseModel
from models.utils import *

class Decoder(BaseModel):
    
    def __init__(self):
        super(Decoder, self).__init__()
        in_channels = 128 + 8
        self.init_dim= 8*8*8*64
        out_channels = self.init_dim
        z_to_dec = []
        z_to_dec.append(
            nn.Sequential(
                nn.Linear(in_channels, out_channels), # paper: 2^18, official repot: 2^15
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        )
        self.z_to_dec = nn.Sequential(*z_to_dec)
        
        # input: B * C * 8 * 8
        # layer1 output: B * 512 * 16 * 16
        # layer2 output: B * 256 * 32 * 32
        # layer3 output: B * 128 * 64 * 64
        # layer4 output: B * 64 * 128 * 128
        # layer5 output: B * 3 * 256 *  256
        in_channels = out_channels // (8*8)
        self.layer_dims = [512, 256, 128, 64, 3]
        self.dropout = [0,1]
        upsample = []
        for i  in range(len(self.layer_dims)-1):
            out_channels = self.layer_dims[i]
            if i in self.dropout:
                upsample.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.Dropout(0.5),
                        nn.ReLU()
                    )
                )
                in_channels = out_channels
            else:
                upsample.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                )
                in_channels = out_channels
        self.upsample = nn.Sequential(*upsample)

        
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.layer_dims[-2], out_channels=self.layer_dims[-1], kernel_size= 4, stride =2, padding = 1),
            nn.Tanh()
        )

        self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             xavier_init(m)


    def forward(self, zx, zs):
        z = torch.cat((zx, zs), dim = 1)
        B, _ = z.size()
        input = self.z_to_dec(z)
        C = self.init_dim // (8*8)
        input = input.view(B, C, 8, 8)
        out = self.upsample(input)
        out = self.last_layer(out)

        return out

