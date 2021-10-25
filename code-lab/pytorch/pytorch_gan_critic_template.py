"""Pytorch basic GAN template for generator critic.

Not fully completed yet. 
To use with generator and discriminator model change critic name to discriminator. 
Output value then should have output value between 0 and 1.

Look into below for more details,
https://github.com/quickgrid/Paper-Implementations/tree/main/pytorch
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


class Critic(nn.Module):
    def __init__(self, img_channels, feature_map_base):
        super(Critic, self).__init__()
        self.C = nn.Sequential(

        )

    def _blocks(self):
        return nn.Sequential(

        )

    def forward(self, x):
        return self.C(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, feature_map_base):
        super(Generator, self).__init__()
        self.G = nn.Sequential(

        )

    def _blocks(self):
        return nn.Sequential(

        )

    def forward(self, z):
        return self.G(z)


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.NUM_WORKERS = 0
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 64
        self.IMAGE_CHANNELS = 3
        self.NUM_EPOCHS = 10
        self.Z_DIM = 100
        self.LEARNING_RATE = 3e-4
        self.GENERATOR_FEATURE_MAP_BASE = 64
        self.CRITIC_FEATURE_MAP_BASE = 64

        gan_dataset = CustomImageDataset(root_dir='', transform=self.get_transform())
        self.train_loader = DataLoader(gan_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

        self.G = Generator(z_dim=self.Z_DIM, img_channels=self.IMAGE_CHANNELS, feature_map_base=self.GENERATOR_FEATURE_MAP_BASE)
        self.C = Critic(img_channels=self.IMAGE_CHANNELS, feature_map_base=self.CRITIC_FEATURE_MAP_BASE)
        self.initialize_weights(self.G)
        self.initialize_weights(self.C)

        # Tensorboard code.
        # Generate tensor directly on device to avoid memory copy.
        # See, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.fixed_noise = torch.randn((32, self.Z_DIM, 1, 1), device=self.device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.step = 0

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(self.IMAGE_CHANNELS)],
                std=[0.5 for _ in range(self.IMAGE_CHANNELS)],
            )
        ])

    def initialize_weights(self, model, mean=0.0, std=0.02):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, mean=mean, std=std)

    def train(self):
        for epoch in range(self.NUM_EPOCHS):
            for batch_idx, (real, _) in enumerate(self.train_loader):
                pass


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
