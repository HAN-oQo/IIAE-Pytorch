import os
import random
from torch.utils import data
from torchvision.datasets.folder import pil_loader

class MNIST_CDCB(data.Dataset):
    def __init__(self, path_to_data =  "/home/hankyu/hankyu/disentangle/iiae/MNIST-CDCB", train = True, transforms = None):
        self.path_to_data = path_to_data
        self.train = train
        self.transforms = transforms

        self.X = []
        self.Y = []

        if self.train:
            self.dataX = os.path.join(self.path_to_data, 'trainX')
            self.dataY = os.path.join(self.path_to_data, 'trainY')
        else:
            self.dataX = os.path.join(self.path_to_data, 'testX')
            self.dataY = os.path.join(self.path_to_data, 'testY')
            
        
        self.prepare()
        print(self.X[0])
        print(self.Y[0])
        print(len(self.X))
        print(len(self.Y))
    def prepare(self):
        for img in os.listdir(self.dataX):
            self.X.append(os.path.join(self.dataX, img))
        for img in os.listdir(self.dataY):
            self.Y.append(os.path.join(self.dataY, img))


    def __getitem__(self, index) :

        X_path = self.X[index % len(self.X)]
        Y_path = self.Y[index % len(self.Y)]

        X_img = pil_loader(X_path)
        Y_img = pil_loader(Y_path)

        if self.transforms is not None:
            X_img = self.transforms(X_img)
            Y_img = self.transforms(Y_img)

        return{"X": X_img, "Y": Y_img}

    def __len__(self):
        return len(self.X) + len(self.Y)          

