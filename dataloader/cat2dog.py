import os
import random
from torch.utils import data
from torchvision.datasets.folder import pil_loader

class Cat2Dog(data.Dataset):

    def __init__(self, path_to_data = '/home/hankyu/hankyu/disentangle/afhq', train = True, transforms = None):

        self.path_to_data = path_to_data
        self.train = train
        self.transforms =  transforms

        self.cats = []
        self.dogs = []

        if self.train:
            self.data_path = os.path.join(self.path_to_data, 'train')
            self.cat_path = os.path.join(self.data_path, 'cat')
            self.dog_path = os.path.join(self.data_path, 'dog')
        else:
            self.data_path = os.path.join(self.path_to_data, 'val')
            self.cat_path = os.path.join(self.data_path, 'cat')
            self.dog_path = os.path.join(self.data_path, 'dog')
        
        self.prepare()
        
    def prepare(self):
        
        for img in os.listdir(self.cat_path):
            self.cats.append(os.path.join(self.cat_path, img))
        for img in os.listdir(self.dog_path):
            self.dogs.append(os.path.join(self.dog_path, img))
    
    def __getitem__(self, index):
        
        cat_path = self.cats[index % len(self.cats)]
        dog_path = self.dogs[random.randint(0, len(self.dogs)-1)]

        cat_img = pil_loader(cat_path)
        dog_img = pil_loader(dog_path)

        if self.transforms is not None:
            cat_img = self.transforms(cat_img)
            dog_img = self.transforms(dog_img)
        
        return {"X": cat_img, "Y": dog_img}
    
    def __len__(self):
        return len(self.cats) + len(self.dogs)


