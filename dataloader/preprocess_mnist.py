import torch
import numpy as np
import os
from torchvision.datasets.folder import pil_loader
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor


data_path = "../MNIST-CDCB"

# data_list = os.listdir(data_path)
# print(len(data_list))
# # for i in range(len(data_list)):

# #     sample_img_path = os.path.join(data_path, data_list[i])
# #     sample_img = pil_loader(sample_img_path)
# #     w, h = sample_img.size
# #     if w != 512 or h != 256:
# #         print("wrong")
# #         print(sample_img_path)

# sample_img_path = os.path.join(data_path, data_list[0])
# sample_img = pil_loader(sample_img_path)
# # a = sample_img.copy()

# a = transforms.ToTensor()(np.array(sample_img))
# print(a.size())
# a1, a2 = torch.split(a, [256,256], dim = -1)
# save_dir = './'
# save_image(a1, os.path.join(save_dir, 'sample1.jpg'))
# save_image(a2, os.path.join(save_dir, 'sample2.jpg'))



def preprocess(path_to_data = data_path):
    train_path = os.path.join(path_to_data, 'train')
    test_path = os.path.join(path_to_data, 'test')

    save_X_train = os.path.join(path_to_data, 'trainX')
    save_Y_train = os.path.join(path_to_data, 'trainY')
    save_X_test = os.path.join(path_to_data, 'testX')
    save_Y_test = os.path.join(path_to_data, 'testY')

    if not os.path.exists(save_X_train):
        os.makedirs(save_X_train)
    if not os.path.exists(save_Y_train):
        os.makedirs(save_Y_train)
    if not os.path.exists(save_X_test):
        os.makedirs(save_X_test)
    if not os.path.exists(save_Y_test):
        os.makedirs(save_Y_test)
    
    for img_path in os.listdir(train_path):
        img = pil_loader(os.path.join(train_path, img_path))
        a = transforms.ToTensor()(np.array(img))
        a1, a2 = torch.split(a, [256,256], dim = -1)
        save_image(a1, os.path.join(save_X_train, img_path))
        save_image(a2, os.path.join(save_Y_train, img_path))
    
    for img_path in os.listdir(test_path):
        img = pil_loader(os.path.join(test_path, img_path))
        a = transforms.ToTensor()(np.array(img))
        a1, a2 = torch.split(a, [256,256], dim = -1)
        save_image(a1, os.path.join(save_X_test, img_path))
        save_image(a2, os.path.join(save_Y_test, img_path))
    
    print(len(os.listdir(save_X_train)))
    print(len(os.listdir(save_Y_train)))
    print(len(os.listdir(train_path)))

    print(len(os.listdir(save_X_test)))
    print(len(os.listdir(save_Y_test)))
    print(len(os.listdir(test_path)))
    
preprocess()
train_X = os.listdir('../MNIST-CDCB/trainX')
train_Y = os.listdir('../MNIST-CDCB/trainY')

for idx in range(len(train_X)):
    if train_X[idx] != train_Y[idx]:
        print("wrong!")
        print(train_X[idx])
        print(train_Y[idx])