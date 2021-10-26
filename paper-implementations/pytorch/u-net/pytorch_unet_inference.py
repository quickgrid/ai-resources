#import cv2
from cv2 import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision

from pytorch_unet_train import UNET


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240


def load_checkpoint(checkpoint, model):
    print('=> LOADING checkpoint.')
    model.load_state_dict(checkpoint["state_dict"])


def segment_extraction(
        original_image_path,
        mask_image_path,
        segment_image_path,
        mode='foreground'
):
    inference_image_mask = np.array(Image.open(mask_image_path).convert("L"), dtype=np.float32)
    inference_image_original = np.array(Image.open(original_image_path).convert("RGB"))
    inference_image_mask = np.asarray(inference_image_mask / 255, dtype=np.uint8)

    inference_image_original_masked = inference_image_original.copy()

    # Mask blur for smoother images.
    inference_image_mask = cv2.GaussianBlur(inference_image_mask, (17, 17), cv2.BORDER_DEFAULT)

    if mode == 'foreground':
        inference_image_original_masked[inference_image_mask == 0] = 0
        inference_image_original_masked[inference_image_mask != 0] = inference_image_original[inference_image_mask != 0]
    elif mode == 'background':
        inference_image_original_masked[inference_image_mask != 0] = 0
        inference_image_original_masked[inference_image_mask == 0] = inference_image_original[inference_image_mask == 0]

    # import matplotlib.pyplot as plt
    # plt.imshow(inference_image_mask)
    # plt.show()

    cv2.imwrite(segment_image_path, cv2.cvtColor(inference_image_original_masked, cv2.COLOR_RGB2BGR))

    return inference_image_original_masked


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

        augmentations = test_transform(image=image)
        retimg = augmentations["image"].unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return retimg.cuda()


    #aimage_path = image_loader(r'data/val_images/0ed6904e1004_12.jpg')
    aimage_path = image_loader(r'data/val_images/0cdf5b5d0ce1_12.jpg')
    #aimage_path = image_loader(r'extra_images/6.jpg')


    folder = 'inference_results'
    inference_img_name = 'inference_img'
    inference_ori_img_name = 'inference_ori_img'
    foreground_img_namge = 'inference_foreground_img'
    background_img_namge = 'inference_background_img'
    foreground_focus = 'foreground_focus_img'

    model.eval()

    with torch.no_grad():
        preds = model(aimage_path)
        preds = (preds > 0.5).float()
        #print(preds)


    save_original_image_path = f"{folder}/inference_{inference_ori_img_name}.png"
    save_mask_image_path = f"{folder}/inference_{inference_img_name}.png"
    save_foreground_image_path = f"{folder}/inference_{foreground_img_namge}.png"
    save_background_image_path = f"{folder}/inference_{background_img_namge}.png"
    save_foreground_focus_image_path = f"{folder}/inference_{foreground_focus}.png"


    torchvision.utils.save_image(
        preds, save_mask_image_path
    )

    torchvision.utils.save_image(
        aimage_path, save_original_image_path
    )

    foreground_image = segment_extraction(
        original_image_path=save_original_image_path,
        mask_image_path=save_mask_image_path,
        segment_image_path=save_foreground_image_path,
        mode='foreground',
    )

    background_image = segment_extraction(
        original_image_path=save_original_image_path,
        mask_image_path=save_mask_image_path,
        segment_image_path=save_background_image_path,
        mode='background',
    )

    # TODO: Smooth extracted foreground borders in saved images.
    foreground_focused_image = cv2.blur(background_image, ksize=(8,8)) + foreground_image
    cv2.imwrite(save_foreground_focus_image_path, cv2.cvtColor(foreground_focused_image, cv2.COLOR_RGB2BGR))