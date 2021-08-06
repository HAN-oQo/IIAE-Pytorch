import os
import random
from torch.utils import data
from torchvision.datasets.folder import pil_loader

class Cars(data.Dataset):
    def __init__(self, path_to_data = "/home/hankyu/hankyu/disentangle/iiae/data/cars", train = True, transforms = None):
        self.path_to_data = path_to_data
        self.train = train
        self.transforms = transforms

        self.X = []
        self.Y = []
        self.items = []
        
        if self.train:
            self.data_path = os.path.join(os.path.join(self.path_to_data, 'train'), '360')
            
        else:
            self.data_path = os.path.join(os.path.join(self.path_to_data, 'test'), '360')
        self.prepare()
        # print(len(self))
        print(self.X[0])
        print(self.Y[0])

    def prepare(self):
        tmp_list = []
        for img_num in os.listdir(self.data_path):
            img_path = os.path.join(self.data_path, img_num)
            img_X_path = os.path.join(img_path, "X")
            img_Y_path = os.path.join(img_path, "Y")
            for i in os.listdir(img_X_path):
                self.X.append(os.path.join(img_X_path, i))
            tmp_list = []
            for i in os.listdir(img_Y_path):
                tmp_list.append(os.path.join(img_Y_path, i))
            self.Y.append(tmp_list)

        print(len(self.X))
        print(len(self.Y))
        for i in range(len(self.Y)):
            if len(self.Y[i]) != 23:
                print("wrong")
                
    def __getitem__(self, index) :

        X_path = self.X[index % len(self.X)]
        Y_path = self.Y[index % len(self.Y)][random.randint(0, len(self.Y[0])-1)]
        
        X_img = pil_loader(X_path)
        Y_img = pil_loader(Y_path)

        if self.transforms is not None:
            X_img = self.transforms(X_img)
            Y_img = self.transforms(Y_img)
        
        return{"X": X_img, "Y": Y_img}

    def __len__(self):
        return len(self.X) + len(self.Y)*len(self.Y[0])            