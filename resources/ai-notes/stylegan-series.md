# [AI Notes](https://github.com/quickgrid/AI-Resources/blob/master/ai-notes.md)

Notes from reading various deep learning, computer vision etc. papers. 

Many of the text are copied from paper as is and others with some modifications. Figures are from paper. Highlighted portions should be read and only some of the highlighted parts are expanded as notes.

**WARNING:** These notes may contain errors due to misinterpretation, lack of understanding, missing details etc. 

# StyleGAN Family

Recommend skimming all of the papers except Style GAN 2 ADA (optional) to get an overall understanding.

# [ProGAN](https://arxiv.org/abs/1710.10196)

**Status: Mostly complete.**

Key idea is growing both Generator `G`, Discriminator `D` progressively. Starting from easier low-resolution images, and add new layers that introduce higher-resolution details as the training progresses.

This incremental nature allows the training to first discover large-scale structure of the image distribution and then shift attention to increasingly finer scale detail, instead of having to learn all scales simultaneously.

It is shown on figure 4 that earlier layers take less time to train. With progressive growing the existing low-resolution layers are likely to have already converged early on,
so the networks are only tasked with refining the representations by increasingly smaller-scale effects as new layers are introduced. Without progressive growing, all layers of the generator and discriminator are tasked with simultaneously finding succinct intermediate representations for both the large-scale variation and the small-scale detail. 

### Background

Generative methods that produce novel samples from high-dimensional data distributions. They include,

- Auto regressive models such as PixelCNN produces sharp result but slow and do not have latent representation. 
- VAE are easy to train but produces blurry results.
- GANs produce sharp images at low resolution with limited variation.

A GAN consists of two networks: `generator` and `discriminator (aka critic)`. The generator produces a sample, such as an image, from a latent code. The distribution of these generated samples such as images should ideally be indistinguishable from the training distribution. A discriminator is trained to do this assesment. Since both networks are differentiable, gradient is used to steer both networks to the right direction.

High resolution image generation is difficult because it easier to distinguish generated samples from real. Also larger samples require using smaller minibatches due to memory cosideration further compromising training quality. 

Several ways were proposed by others to measure degrees of variation in generative model such as, `Multi-Scale Structural Similarity (MS-SSIM)`, `Inception Score (IS)`.


### Evaluation Metric

MS-SSIM is able to find large-scale `mode collapses` reliably but fail to react to smaller effects such as loss of variation in colors or textures, and they also do not directly assess image quality in terms of similarity to the training set.

Patches are extracted based on section 5 for which statistical similarity is measured by computing their `sliced Wasserstein distance (SWD)`. 

A small Wasserstein distance indicates that the distribution of the patches is similar, meaning that the training images and generator samples appear similar in both appearance and variation at this spatial resolution. The distance between the patch sets extracted from the lowest resolution images indicate similarity in large-scale image structures.


## Network Structure

![alt text](figures/progan/progan1.png)


## Progressive Growing

![alt text](figures/progan/progan5.png)


## Layer Fading

![alt text](figures/progan/progan3.png)



## Implementation Details

All layers of both network remain trainable throught training process and newly added layers are fade in smoothly. Both G and D are mirrors of each other and grow in synchrony.


![alt text](figures/progan/progan2.png)


### Network Details

As shown in above diagram the 3 layer blocks are repeated multiple time for both network. For G it is `(upsample, conv, conv)` and for D it is `(conv, conv, downsample)`. This will be block that is reused in code to generate the repeated layers.

The last conv block is `toRGB` in generator. Uses `1x1` convolution to map feature map to 3 channels for RGB image. Similarly for discriminator the first block with `1x1` convolution is `fromRGB` which takes an RGB 3-channel image and maps to desired feature map size.

Upsampling uses `2x2` replication and downsampling is `2x2` average pooling. As shown in table 2 all layers use leaky relu with negative slope of `0.2` except last layer using linear activation.

No batch, layer, weight norm is used but after each `3x3` conv layer in `G` pixel norm is used.

#### Pixelwise Normalization

Pixelwise feature vector normalization aka `pixel norm` is applied in generator after each conv layer to prevent `G` and `D` magnitute spiral out of control. Each pixel in channel/feature map dimension is normalized using simple formula in section 4.2.


![alt text](figures/progan/progan6.png)



### Training Details

G and D optimization is alternate on per minibatch basis.

Training start with `4x4` resolution. Latent vector `z` is 512 dimensional and images are normalized to `[-1, 1]`. 

![alt text](figures/progan/progan7.png)

### Loss Function

WGAN-GP.

### Weight Initialization

Weight initialization is performed with bias set to 0 and all weights from normal distribution with unit variance. Weights are initialized based on `he/kaiming initializer`. For pytorch it is `kaiming_normal_`. 




## Dataset Generation (Optional)

In this paper a higher quality version 1024x1024 CelebA dataset with 30000 images is used.

To improve the overall image quality, JPEG images were processed with two pre-trained neural networks: a convolutional autoencoder trained to remove JPEG artifacts in natural images and an adversarially-trained 4x super-resolution network. 

Based on face landmarks a rotated bounding box is selected and it is then orientated, cropped to generate training image.

![alt text](figures/progan/progan4.png)

<hr>

# [StyleGAN](https://arxiv.org/abs/1812.04948)

An `alternate generator structure` is proposed that leads unsupervised separation of high level attributes such as pose, identity for faces, `stochastic variation` such as freckles, hair etc. The `discriminator, loss function stays same` from previous work.

New generator `starts from a learned constant input` and adjusts the `style` of the image at each convolution layer based on the latent code, therefore directly controlling the strength of image features at different scales.

The input latent space must follow the probability density of the training data. It is said that this leads to some degree of unavoidable entanglement. To overcome this generator embeds latent code into intermediate space. This intermediate space is free from restriction thus allowed to be disentangled.

For quantifying the amount of disentanglement in latent space `Perceptual Path Length` and `Linear Separability` is proposed to quantify these aspects in generator.

It is shown that new generator gets more linear, less entangled representation of different factors of variation than previous. Also a high quality human faces dataset `FFHQ` is proposed.

![alt text](figures/stylegan/stylegan1.png)

![alt text](figures/stylegan/stylegan2.png)

![alt text](figures/stylegan/stylegan3.png)

# [StyleGAN 2](https://arxiv.org/abs/1912.04958)

![alt text](figures/stylegan2/stylegan2-1.png)

![alt text](figures/stylegan2/stylegan2-2.png)

![alt text](figures/stylegan2/stylegan2-3.png)

![alt text](figures/stylegan2/stylegan2-4.png)

# [StyleGAN 2 ADA](https://arxiv.org/abs/2006.06676)

![alt text](figures/stylegan2-ada/stylegan2-ada-1.png)

![alt text](figures/stylegan2-ada/stylegan2-ada-2.png)

![alt text](figures/stylegan2-ada/stylegan2-ada-3.png)

# [StyleGAN 3](https://arxiv.org/abs/2106.12423)

![alt text](figures/stylegan3/stylegan3-1.png)

![alt text](figures/stylegan3/stylegan3-2.png)

