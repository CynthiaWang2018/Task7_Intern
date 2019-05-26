import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class TaskDataset(Dataset):

    def __init__(self, x_path, label_path):
        self.x_path = x_path
        self.x = []
        self.y = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                img_path, label = line.split()
                self.x.append(img_path)
                self.y.append(int(label))

        self.y = np.array(self.y, dtype=np.float32)

    def __getitem__(self, index):
        img = np.array(Image.open(self.x_path+self.x[index]), dtype=np.float32) / 255.
        label = self.y[index]

        return img, label  # [32, 32]

    def __len__(self):
        return len(self.x)

class TaskDataset2(Dataset):

    def __init__(self, x_path):
        self.x_path = x_path
        self.x = os.listdir(x_path)

    def __getitem__(self, index):
        img = np.array(Image.open(self.x_path+self.x[index]), dtype=np.float32) / 255.
        return img

    def __len__(self):
        return len(self.x)