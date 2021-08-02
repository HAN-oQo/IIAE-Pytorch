import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from models import BaseModel
from models.utils import *

class Exclusive_Specific_Encoder(BaseModel):

    def __init__(self):
        super(Exclusive_Specific_Encoder, self).__init__()

        # input: Batch_size * 3 * 256 * 256
        self.layer_dims = [32, 64, 128, 256]
        self.mu_dims = 8
        self.latent_dims = self.mu_dims * 2
        
        # in: B * 3 * 256 * 256
        # layer 1 out: B * 32 * 128 * 128
        # layer 2 out: B * 64 * 64 * 64
        # layer 3 out: B * 128 * 32 * 32
        # layer 4 out: B * 256 * 16 * 16

        downsample = []
        in_channels = 3
        for out_channels in self.layer_dims: 
            downsample.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size= 4, stride= 2, padding= 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.02)
                )
            )
            in_channels = out_channels
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(nn.Linear(self.layer_dims[-1]* 16 * 16, self.latent_dims))

        self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             xavier_init(m)

    # def reparameterize(self, mu, log_var):

    #     std = torch.exp(0.5 * log_var)s
    #     eps = torch.randn_like(std)
    #     return mu + (eps * std)
        
    def forward(self, input):
        B, _, _, _ = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)
        return [mu, log_var, z]

class Shared_Feature_extractor(BaseModel):
    def __init__(self):
        super(Shared_Feature_extractor, self).__init__()
        self.layer_dims = [32, 64, 128, 256]
        downsample = []
        in_channels = 3
        for out_channels in self.layer_dims: 
            downsample.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size= 4, stride= 2, padding= 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.02)
                )
            )
            in_channels = out_channels
        self.downsample = nn.Sequential(*downsample)
        self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             xavier_init(m)

    def forward(self, input):
        out = self.downsample(input)
        return out

class Exclusive_Shared_Encoder(BaseModel):
    def __init__(self):
        super(Exclusive_Shared_Encoder, self).__init__()

        self.mu_dims = 128
        self.latent_dims = self.mu_dims * 2

        downsample = []
        in_channels = 256
        out_channels = 256
        downsample.append(
            nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size= 5, stride= 1, padding= 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02)
            )
        )
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(
            nn.Linear(out_channels*16*16, self.latent_dims) , 
            nn.Linear(self.latent_dims, self.latent_dims)
        )

        self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             xavier_init(m)

    # def reparameterize(self, mu, log_var):
    
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return mu + (eps * std)

    def forward(self, input):
        B, _, _, _ = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)
        return [mu, log_var, z]


class Common_Shared_Encoder(BaseModel):
    
    def __init__(self):
        super(Common_Shared_Encoder, self).__init__()
        self.mu_dims = 128
        self.latent_dims = self.mu_dims * 2

        downsample = []
        in_channels = 256 * 2
        out_channels = 256
        downsample.append(
            nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size= 5, stride= 1, padding= 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02)
            )
        )
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(
            nn.Linear(out_channels*16*16, self.latent_dims),
            nn.Linear(self.latent_dims, self.latent_dims)
        )

        self.weight_init()

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             xavier_init(m)

    # def reparameterize(self, mu, log_var):
    
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return mu + (eps * std)

    def forward(self, inputX, inputY):
        input = torch.cat((inputX, inputY), dim = 1) #channel-wise concat
        B, _, _, _ = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)

        return [mu, log_var, z]


