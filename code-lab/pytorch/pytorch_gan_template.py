import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self):
        pass


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(CustomImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(image_path)
        return self.transform(image)    
    

class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self):
        pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    
