# IIAE(Interaction Information AutoEncoder) - Pytorch Implementation

Pytorch Implementation of [Variational Interaction Information Maximization for Cross-domain Disentanglement(IIAE)](https://proceedings.neurips.cc/paper/2020/file/fe663a72b27bdc613873fbbb512f6f67-Paper.pdf)

This implementation is focused on Image-to-Image translation task of IIAE.


Official Repository(tensorflow implementation) is [here](https://github.com/gr8joo/IIAE).


<p align="center"><img width="100%" src="imgs/IIAE.png" /></p>

---
## Dependencies
To resolve the dependencies manually, install
- python >= 3.6
- torch==1.7.1+cu110
- torchvision==0.8.2+cu110
- json
- tensorboard

Or if you are using Anaconda, simply run command
```
# create same conda environment

conda env create -f requirements.txt
```
---
## Datasets
### Datasets
1. MNIST-CDCB
2. Cars
3. AFHQ (Stil working for high quality result)

### Download datasets
for MNIST-CDCB, 
```
# Download MNIST-CDCB dataset
sudo bash download_mnist_cbcd.sh
```

for Cars,
```
# Download Cars dataset
sudo bash download_mnist_Cars.sh
```
for AFHQ, (download.sh file is from [stargan v2 official repository](https://github.com/clovaai/stargan-v2))

```
# Download AFHQ dataset
sudo bash download.sh afhq-dataset
```

### Preprocess datasets

MNIST-CDCB dataset and Cars dataset need to be preprocessed.
You can preprocess the datasets for the network by running the codes in 'dataloader' directory

```
# for MNIST-CDCB,
python preprocess_mnist.py

# for AFHQ,
python preprocess_cars.py
```

When you try to run the preprocessing code, permission error can be occured.
The error can be resolved by this command.
```
sudo chmod 0777 <directory>
```
---
## Train / Test Network

With following command and config file, you can train and test the network.

```
python main.py <config file path>
```
### Config 
Here is the sample config file. Please follow the template.  
Set **train: 1** for train, **train: 0** for test.

```
{
    "id": "IIAE_MNIST",
    "model": "IIAE",
    "dataset": "MNIST-CDCB",
    "path_to_data": "./MNIST-CDCB",
    "train" : 1,  
    "resolution": 256,
    "training": {
        "max_iters": 50000,
        "resume_iters": 0,
        "restored_model_path": "",
        "batch_size": 16,
        "lr": 0.00002,
        "weight_decay": 0.0,
        "beta1" : 0.9,
        "beta2" : 0.999,
        "milestones": [10000, 30000], 
        "scheduler_gamma": 0.5,
        "rec_weight": 100.02,
        "X_kld_weight": 5.02,
        "S_kld_weight": 0.02,
        "inter_kld_weight": 5,
        "print_freq": 100,
        "sample_freq": 2000,
        "model_save_freq": 5000
    },
    "test":{
        "test_iters": 50000,
        "batch_size" : 16,
        "test_path": "./2021-08-06_22-14_IIAE_MNIST"
    }
}
```

---
## Sample  Results
### 1. MNIST-CDCB

### 2. Cars

### 3. AFHQ

## Pre-trained Models
### 1. [MNIST-CDCB(50000 iterations)]()
### 2. [Cars(50000 iterations)]()
### 3. [AFHQ]()