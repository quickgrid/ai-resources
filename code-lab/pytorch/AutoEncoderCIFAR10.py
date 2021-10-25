"""
References:
https://www.kaggle.com/ljlbarbosa/convolution-autoencoder-pytorch
"""

import gc

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(17)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, latent_channel_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=latent_channel_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.pool1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channel_dim):
        super(Decoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(in_channels=latent_channel_dim, out_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2))
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.act1(x)
        x = self.t_conv2(x)
        x = self.act2(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_channel_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_channel_dim=latent_channel_dim).to(device)
        self.decoder = Decoder(latent_channel_dim=latent_channel_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def test_autoencoder(input_data_shape, latent_channel_dim):
    random_data = torch.randn(input_data_shape).to(device)
    autoencoder = AutoEncoder(latent_channel_dim=latent_channel_dim).to(device)
    result = autoencoder(random_data)
    print(result.shape)

    del random_data
    del autoencoder

    assert result.shape == input_data_shape, F"Output of decoder must match original image dimensions."


def test_decoder_output_channels(input_data_shape, latent_channel_dim):
    random_data = torch.randn(input_data_shape).to(device)
    encoder1 = Encoder(latent_channel_dim=latent_channel_dim).to(device)
    result = encoder1(random_data)
    print(result.shape)

    random_data = torch.randn(result.shape).to(device)
    decoder1 = Decoder(latent_channel_dim= latent_channel_dim).to(device)
    result = decoder1(random_data)
    print(result.shape)

    del random_data
    del encoder1
    del decoder1

    assert result.shape == input_data_shape, F"Output of decoder must match original image dimensions."


def test_encoder_output_channels(input_data_shape, latent_channel_dim):
    random_data = torch.randn(input_data_shape).to(device)
    encoder1 = Encoder(latent_channel_dim=latent_channel_dim).to(device)
    result = encoder1(random_data)
    print(result.shape)

    del random_data
    del encoder1

    assert result.shape[1] == latent_channel_dim, F"Channel dimension should be {latent_channel_dim}."


def print_model_details(model):
    print(model)


def test_models():
    latent_channel_dim = 4

    test_encoder_output_channels(input_data_shape=(1, 3, 28, 28), latent_channel_dim=latent_channel_dim)
    test_encoder_output_channels(input_data_shape=(8, 3, 200, 200), latent_channel_dim=latent_channel_dim)

    test_decoder_output_channels(input_data_shape=(1, 3, 28, 28), latent_channel_dim=latent_channel_dim)
    test_decoder_output_channels(input_data_shape=(8, 3, 200, 200), latent_channel_dim=latent_channel_dim)

    test_autoencoder(input_data_shape=(1, 3, 28, 28), latent_channel_dim=latent_channel_dim)
    test_autoencoder(input_data_shape=(8, 3, 200, 200), latent_channel_dim=latent_channel_dim)

    print_model_details(AutoEncoder(latent_channel_dim))

    gc.collect()
    torch.cuda.empty_cache()

    print(gc.get_count())
    print(gc.get_stats())




def main():
    latent_channel_dim = 4
    n_epochs = 100
    num_workers = 0
    batch_size = 32

    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])
    plt.show()

    model = AutoEncoder(latent_channel_dim).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        train_loss = 0.0

        chosen_output = None
        chosen_image = None

        for data in train_loader:
            images, _ = data
            images = images.to(device)

            # Zero out previous grads, forward pass, calculate loss, backpropagate, optimize
            optimizer.zero_grad()
            outputs = model(images)

            chosen_output = outputs[0]
            chosen_image = images[0]

            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))

        # Training is blocked on non notebook until window closed
        if epoch % 10 == 0:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.transpose(chosen_output.detach().cpu().numpy(), (1, 2, 0)))
            ax[1].imshow(np.transpose(chosen_image.detach().cpu().numpy(), (1, 2, 0)))
            plt.show()


if __name__ == '__main__':
    main()
