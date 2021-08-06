import torch
import numpy as np
import os
import scipy.io
# from scipy.misc import imresize
import pandas as pd
import cv2
import torch
from torchvision.utils import save_image
# from scipy.io import loadmat

car_path = "../data/cars/"


def preprocess(test=False, ver=360, interval=1, half=None, angle_info=False, image_size=128, gray=True):
    # car_files = map(lambda x: os.path.join(car_path, x), os.listdir( car_path ))
    # car_files = filter(lambda x: x.endswith('.mat'), car_files)

    # car_idx = map(lambda x: int(x.split('car_')[1].split('_mesh')[0]), car_files )
    # car_df = pd.DataFrame( {'idx': car_idx, 'path': car_files}).sort_values(by='idx')

    # car_files = car_df['path'].values
    car_list = os.listdir(car_path)
    car_files = []
    for i in range(len(car_list)-1):
        car_files.append(os.path.join(car_path, car_list[i]))
    print(len(car_files))
    
    if not test:
        car_files = car_files[:-14]
    else:
        car_files = car_files[-14:]
    
    print(len(car_files))
    car_images = []
    classes = []

    n_cars = len(car_files)
    car_idx = 0
    for car_file in car_files:
        if not car_file.endswith('.mat'): continue
        car_mat = scipy.io.loadmat(car_file)
        car_ims = car_mat['im']
        car_idx += 1

        if half == 'first':
            if car_idx > n_cars / 2:
                break
        elif half == 'last':
            if car_idx <= n_cars / 2:
                continue

        if ver == 360:
            for idx,i in enumerate(list(range(24))):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                if gray:
                    car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
                    car_image = np.repeat(car_image[:,:,None], 3, 2)
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
                if angle_info:
                    classes.append(idx)

        elif ver == 180:
            for idx,i in enumerate(list(range(5,-1,-1)) + list(range(23,18,-1))):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                if gray:
                    car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
                    car_image = np.repeat(car_image[:,:,None], 3, 2)
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
        if angle_info:
            classes.append( idx )

        elif ver == 90:
            for idx,i in enumerate(list(range(5,-1,-1))):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
                if angle_info:
                    classes.append( idx )

    car_images = car_images[::interval]
    if test:   
        save_dir = os.path.join(os.path.join(car_path, 'test'), str(ver))
        if not os.path.exists(save_dir):
            os.umask(0)
            os.makedirs(save_dir)
    else:
        save_dir = os.path.join(os.path.join(car_path, 'train'), str(ver))
        if not os.path.exists(save_dir):
            os.umask(0)
            os.makedirs(save_dir)

    # print(len(car_images))
    # print(car_images[0].shape)
    if ver == 360:
        for idx in range(len(car_images)//24):
            save_idx_dir = os.path.join(os.path.join(save_dir, str(idx)))
            if not os.path.exists(save_idx_dir):
                os.makedirs(save_idx_dir)
            save_X_dir = os.path.join(os.path.join(save_idx_dir, 'X'))
            save_Y_dir = os.path.join(os.path.join(save_idx_dir, 'Y'))
            if not os.path.exists(save_X_dir):
                os.makedirs(save_X_dir)
            if not os.path.exists(save_Y_dir):
                os.makedirs(save_Y_dir)
            for num in range(24):
                if num == 0:
                    img_ten = torch.tensor(car_images[idx*24 + num], dtype=torch.float32)
                    save_image(img_ten, os.path.join(save_X_dir, 'car{}.jpg'.format(num)))
                else:
                    img_ten = torch.tensor(car_images[idx*24 + num], dtype=torch.float32)
                    save_image(img_ten, os.path.join(save_Y_dir, 'car{}.jpg'.format(num)))


    # if angle_info:
    #     return np.stack(car_images), np.array(classes)

    # return np.stack( car_images )

preprocess()