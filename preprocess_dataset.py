from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

class preprocess_fer(data.Dataset):
    def __init__(self, split='Training'):
        self.split = split  
        self.data = h5py.File('./data/data.h5', 'r', driver='core')
        if self.split == 'Training':
            self.train_data = np.array(self.data['Training_pixel']).reshape((28709, 48, 48))
            self.train_labels = self.data['Training_label']
        elif self.split == 'PublicTest':
            self.PublicTest_data = np.array(self.data['PublicTest_pixel']).reshape((3589, 48, 48))
            self.PublicTest_labels = self.data['PublicTest_label']
        else:
            self.PrivateTest_data = np.array(self.data['PrivateTest_pixel']).reshape((3589, 48, 48))
            self.PrivateTest_labels = self.data['PrivateTest_label']
        self.transform_train = transforms.Compose([transforms.RandomCrop(44),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])
        self.transform_test = transforms.Compose([transforms.TenCrop(44),
             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    
    def __getitem__(self, index):
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.split == 'Training':
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)