"""Pytorch UNET training implementation.

Notes:
    - Make the implementation simplified and organize codes.

References:
    - UNET implementation tutorial, https://www.youtube.com/watch?v=IHq1t7NxS8k&t=22s.
    - UNET code, https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet.
"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.conv(X)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path of UNET.
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Upward path of UNET.
        # Transposed convolution is used to scale up image before applying double convolutions.
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=(2,2), stride=(2,2))
            )
            self.ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1] * 2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, X):
        skip_connections = []

        # Create connections in downward path.
        for down in self.downs:
            X = down(X)
            skip_connections.append(X)
            X = self.pool(X)

        X = self.bottleneck(X)

        # Reverse the connections such that the last added skip connection is at 0th position.
        skip_connections = skip_connections[::-1]

        # Upward connection of network to connect skip connections with upsample image.
        # In ups list in odd positions are upsample ones and in even positions are double convolutions.
        for idx in range(0, len(self.ups), 2):
            X = self.ups[idx](X)
            skip_connection = skip_connections[idx // 2]

            # In case image shape is odd and rounded to integer then skip connection and X will not match.
            # In this case the input is resized to match shape for concatenation.
            if X.shape != skip_connection.shape:
                X = F.resize(X, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, X), dim=1)
            X = self.ups[idx + 1](concat_skip)

        return self.final_conv(X)


class Tester():
    def __init__(self):
        super(Tester, self).__init__()

    def unet_test(self):
        X = torch.randn((3, 1, 161, 161))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(X)
        print(X.shape)
        print(preds.shape)
        assert X.shape == preds.shape, "Shape of input and predicted images did not match."


class CarvanaDataset(Dataset):
    def __init__(
            self,
            image_dir,
            mask_dir,
            transform=None
    ):
        super(CarvanaDataset, self).__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])

        # Since train image and mask have similar names, but different extensions
        # the masks can be loaded by replacing '.jpg' with mask ending name.
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(image_path).convert("RGB"))

        # Masks are grayscale single channels images.
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Since masks are black and white with value of 0 and 255.
        # These values can be normalized by replacing white value of 255 to 1.
        mask[mask == 255.0] = 1.0

        # Apply augmentation is available.
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = True
LOAD_MODEL = False
# Smaller height, width used for faster calculation.
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
TRAIN_IMG_DIR = 'data/train_images'
TRAIN_MASK_DIR = 'data/train_masks'
VAL_IMG_DIR = 'data/val_images'
VAL_MASK_DIR = 'data/val_masks'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass.
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # tqdm update loss.
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # Use cross entropy loss for multiple classes.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_carvana.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model.
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Check accuracy.
        check_accuracy(val_loader, model, device=DEVICE)

        # Print some predictions to folder.
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )



def save_checkpoint(state, filename='unet_carvana.pth.tar'):
    print('=> Saving checkpoint.')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('=> Saving checkpoint.')
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        train_img_dir,
        train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = CarvanaDataset(
        val_img_dir,
        val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


# Copied as is from code.
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder='saved_images/', device='cuda'
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )

        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}{idx}.png"
        )
    model.train()


if __name__ == "__main__":
    Tester().unet_test()
    main()