import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from pytorch_unet_implementation import UNET


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240


def load_checkpoint(checkpoint, model):
    print('=> LOADING checkpoint.')
    model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("unet_carvana.pth.tar"), model)

    # print(model)

    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )


    def image_loader(image_path):
        """load image, returns cuda tensor"""
        image = np.array(Image.open(image_path).convert("RGB"))

        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        #image = loader(image).float()
        augmentations = test_transform(image=image)
        retimg = augmentations["image"].unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        #retimg = augmentations["image"].unsqueeze(1)
        #return retimg  # assumes that you're using GPU
        return retimg.cuda()


    aimage_path = image_loader(r'data/val_images/0ed6904e1004_12.jpg')
    #aimage_path = image_loader(r'data/val_images/0cdf5b5d0ce1_12.jpg')
    #aimage_path = image_loader(r'extra_images/6.jpg')


    folder = 'inference_results'
    inference_img_name = 'inference_img'
    inference_ori_img_name = 'inference_ori_img'

    model.eval()

    with torch.no_grad():
        preds = model(aimage_path)
        preds = (preds > 0.5).float()
        #print(preds)

    torchvision.utils.save_image(
        preds, f"{folder}/inference_{inference_img_name}.png"
    )

    torchvision.utils.save_image(
        aimage_path, f"{folder}/inference_{inference_ori_img_name}.png"
    )

