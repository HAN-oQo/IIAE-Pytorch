import random
import torch
from torch.utils import data
from torchvision import transforms
from dataloader.cat2dog import Cat2Dog

def load_dataloader(dataset='AFHQ', path_to_data= '/home/hankyu/hankyu/disentangle/afhq', train= True, size= 256, batch_size = 8):
    
    prob = 0.5
    
    if train:
        crop = transforms.RandomResizedCrop(size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
        rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < prob else x)
        all_transforms = transforms.Compose([
            rand_crop,
            transforms.Resize([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])
    
    else:
        all_transforms = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])
    Data_Set = Cat2Dog(path_to_data= path_to_data, train = train, transforms = all_transforms)
    print(len(Data_Set))
    data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = True,  num_workers = 0)

    return data_loader, len(Data_Set)