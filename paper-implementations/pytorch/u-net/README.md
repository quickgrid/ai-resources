## Notes

The model was trained on `Carvana` dataset. A checkpoint is saved per epoch as well as some predicted, its corresponding real image masks. These are saved in `saved_images` folder. 

Using `3 epochs` and `IMAGE_HEIGHT = 160, IMAGE_WIDTH = 240` it was enough to get decent result on the dataset. Validaiton input images and masks were created by taking first `300` images from both train image and mask folder. Train model size is `~360mb`.

`Albumentation` must be installed via pip and if there is any error then upgrading numpy may solve it.

## Results

Some results are not good because these were collected from web and these types of images were not in the dataset.

![Mask](results/a1.png "Mask")
![Image](results/a2.png "Image")
![Foreground](results/a3.png "Foreground")
![Background](results/a4.png "Background")

<br>

![Mask](results/b1.png "Mask")
![Image](results/b2.png "Image")
![Foreground](results/b3.png "Foreground")
![Background](results/b4.png "Background")

<br>

![Mask](results/c1.png "Mask")
![Image](results/c2.png "Image")
![Foreground](results/c3.png "Foreground")
![Background](results/c4.png "Background")

<br>

![Mask](results/d1.png "Mask")
![Image](results/d2.png "Image")
![Foreground](results/d3.png "Foreground")
![Background](results/d4.png "Background")

<br>

![Mask](results/e1.png "Mask")
![Image](results/e2.png "Image")
![Foreground](results/e3.png "Foreground")
![Background](results/e4.png "Background")


## TODO

- Make the implementation simplified.
- Organize codes.

## References

- U-NET implementation tutorial, https://www.youtube.com/watch?v=IHq1t7NxS8k&t=22s.
- U-NET code, https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet.
