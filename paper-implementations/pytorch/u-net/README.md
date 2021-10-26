## Notes

The model was trained on `Carvana` dataset. A checkpoint is saved per epoch as well as some predicted, its correcsponding real image masks. These are saved in `saved_images` folder. 

Using `3 epochs` and `IMAGE_HEIGHT = 160, IMAGE_WIDTH = 240` it was enough to get decent result on the dataset. `Albumentation` must be installed via pip and if there is any error then upgrading numpy may solve it.

## Results



## References

- U-NET implementation tutorial, https://www.youtube.com/watch?v=IHq1t7NxS8k&t=22s.
- U-NET code, https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet.
