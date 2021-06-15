"""Pytorch WGAN implementation with paper references.

This code builds on top of previous GAN and DCGAN code.
Code structure tries to follow these repositories,
- https://github.com/lucidrains/lightweight-gan/.
- https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py.

Notes:
    - Look into using pathlib for file related tasks.

References:
    - WGAN and WGAN-GP implementation, https://www.youtube.com/watch?v=pG0QZ7OddX4.
    - WGAN paper, https://arxiv.org/abs/1701.07875.
    - WGAN GP paper, https://arxiv.org/abs/1704.00028.
    - DCGAN implementation, https://www.youtube.com/watch?v=IZtv9s_Wx9I.
    - DCGAN paper, https://arxiv.org/abs/1511.06434.
    - GAN implementation, https://www.youtube.com/watch?v=OljTVUVzPpM.
    - GAN paper, https://arxiv.org/abs/1406.2661.
    - WGAN section, https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/.
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

        def _blocks(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.LayerNorm(out_channels),
                nn.LeakyReLU(0.2)
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)

        self.C = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_base, k_size, s_amount, p_amount),
            nn.LeakyReLU(0.2),
            *_blocks(feature_map_base, feature_map_base * 2, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 2, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 8, k_size, s_amount, p_amount),
            nn.Conv2d(feature_map_base * 8, 1, k_size, s_amount, padding=(0, 0))
        )

    def forward(self, x):
        return self.C(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, feature_map_base):
        super(Generator, self).__init__()

        def _blocks(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        k_size = (4, 4)
        s_amount = (2, 2)
        p_amount = (1, 1)

        self.G = nn.Sequential(
            *_blocks(z_dim, feature_map_base * 16, k_size, stride=(1, 1), padding=(0, 0)),
            *_blocks(feature_map_base * 16, feature_map_base * 8, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 8, feature_map_base * 4, k_size, s_amount, p_amount),
            *_blocks(feature_map_base * 4, feature_map_base * 2, k_size, s_amount, p_amount),
            nn.ConvTranspose2d(feature_map_base * 2, img_channels, k_size, s_amount, p_amount),
            nn.Tanh()
        )

    def forward(self, z):
        return self.G(z)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, image_size, image_channels):
        super(CustomImageDataset, self).__init__()
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(image_channels)],
                std=[0.5 for _ in range(image_channels)],
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(image_path)
        return self.transform(image)


class Trainer():
    def __init__(
            self,
            root_dir='',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            num_workers=0,
            batch_size=64,
            image_size=64,
            image_channels=3,
            num_epochs=10,
            z_dim=100,
            learning_rate=1e-4,
            generator_feature_map_base=64,
            critic_feature_map_base=64,
            critic_iterations=5,
            lambda_gp=10,
            *args,
            **kwargs
    ):
        super(Trainer, self).__init__()

        self.NUM_EPOCHS = num_epochs
        self.DEVICE = device
        self.CRITIC_ITERATIONS = critic_iterations
        self.LAMBDA_GP = lambda_gp
        self.BATCH_SIZE = batch_size
        self.Z_DIM = z_dim

        gan_dataset = CustomImageDataset(root_dir=root_dir, image_size=image_size, image_channels=image_channels)
        self.train_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        def _initialize_weights(model, mean=0.0, std=0.02):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                    nn.init.normal_(m.weight.data, mean=mean, std=std)

        self.G = Generator(z_dim=z_dim, img_channels=image_channels, feature_map_base=generator_feature_map_base).to(self.DEVICE)
        self.C = Critic(img_channels=image_channels, feature_map_base=critic_feature_map_base).to(self.DEVICE)
        _initialize_weights(self.G)
        _initialize_weights(self.C)

        self.G.train(True)
        self.C.train(True)

        self.opt_G = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        self.opt_C = optim.Adam(self.C.parameters(), lr=learning_rate, betas=(0.0, 0.9))

        # Tensorboard code.
        # Generate tensor directly on device to avoid memory copy.
        # See, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html.
        self.fixed_noise = torch.randn((32, z_dim, 1, 1), device=device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
        self.step = 0

    def get_gradient_penalty(self, real, fake):
        BATCH_SIZE, C, H, W = real.shape
        epsilon = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.DEVICE).repeat(1, C, H, W)
        interpolated_images = real * epsilon + fake * (1 - epsilon)

        mixed_scores = self.C(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def train(self):
        for epoch in range(self.NUM_EPOCHS):
            for batch_idx, (real) in enumerate(self.train_loader):
                real = real.to(self.DEVICE)
                current_batch_size = real.shape[0]

                # Critic optimization.
                mean_iteration_critic_loss = 0
                for _ in range(self.CRITIC_ITERATIONS):
                    noise = torch.randn((current_batch_size, self.Z_DIM, 1, 1), device=self.DEVICE)
                    fake = self.G(noise)
                    critic_real = self.C(real).reshape(-1)
                    critic_fake = self.C(fake.detach()).reshape(-1)
                    gradient_penalty = self.get_gradient_penalty(real=critic_real, fake=critic_fake)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.LAMBDA_GP * gradient_penalty
                    mean_iteration_critic_loss += loss_critic.item() / self.CRITIC_ITERATIONS
                    self.opt_C.zero_grad()
                    loss_critic.backward()
                    self.opt_C.step()

                # Generator optimization.
                noise = torch.randn((current_batch_size, self.Z_DIM, 1, 1), device=self.DEVICE)
                fake = self.G(noise)
                output = self.C(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                self.opt_G.zero_grad()
                loss_gen.backward()
                self.opt_G.step()

                if batch_idx % 100 == 0:
                    self.G.eval()
                    self.C.eval()
                    print(
                        f"EPOCH: [{epoch} / {self.NUM_EPOCHS}], BATCH: [{batch_idx} / {len(self.train_loader)}], "
                        f"LOSS(Mean) D: {mean_iteration_critic_loss:.4f}, LOSS G: {loss_gen:.4f}"
                    )
                    with torch.no_grad():
                        fake = self.G(self.fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                        self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                        self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)
                    self.step += 1
                    self.G.train(True)
                    self.C.train(True)


class GANUtils():
    def __init__(self):
        super(GANUtils, self).__init__()
        self.model_args = dict(
            root_dir='C:\\Users\\computer\Desktop\\celeba gan\\celeba_hq_dataset\\images',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            NUM_WORKERS=0,
            BATCH_SIZE=64,
            IMAGE_SIZE=64,
            IMAGE_CHANNELS=3,
            NUM_EPOCHS=10,
            Z_DIM=100,
            LEARNING_RATE=1e-4,
            GENERATOR_FEATURE_MAP_BASE=64,
            CRITIC_FEATURE_MAP_BASE=64,
            CRITIC_ITERATIONS=5,
            LAMBDA_GP=0.01
        )

    def get_train_config(self, debug=True):
        if debug:
            print(self.model_args)
        return self.model_args


if __name__ == '__main__':
    trainer = Trainer(**(GANUtils().get_train_config()))
    trainer.train()
