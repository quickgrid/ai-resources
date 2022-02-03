# AI Notes

Notes from reading various deep learning, computer vision etc. papers. Many of the text are copied from paper as is and others with some modifications. Figures are from paper.

**WARNING:** These notes may contain errors due to misinterpretation, lack of understanding, missing details etc. 

# Style GAN Family

Recommend skimming all of the papers except Style GAN 2 ADA (optional) to get an overall understanding.

## ProGAN

Key idea is growing both generator, discriminator progressively. Starting from easier low-resolution images, and add new layers that introduce higher-resolution details as the training progresses.

#### Background

Generative methods that produce novel samples from high-dimensional data distributions. They include,

- Auto regressive models such as PixelCNN produces sharp result but slow and do not have latent representation. 
- VAE are easy to train but produces blurry results.
- GANs produce sharp images at low resolution with limited variation.

A GAN consists of two networks: `generator` and `discriminator (aka critic)`. The generator produces a sample, such as an image, from a latent code. The distribution of these generated samples such as images should ideally be indistinguishable from the training distribution. A discriminator is trained to do this assesment. Since both networks are differentiable, gradient is used to steer both networks to the right direction.

High resolution image generation is difficult because it easier to distinguish generated samples from real. Also larger samples require using smaller minibatches due to memory cosideration further compromising training quality. 

Several ways were proposed by others to measure degrees of variation in generative model such as, `Multi-Scale Structural Similarity (MS-SSIM)`, `Inception Score (IS)`.

### Network Structure

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/progan/progan1.png)

### Implementation Details

They use improved wasserstein loss.

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/progan/progan2.png)

### Progressive Growing

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/progan/progan5.png)

### Layer Fading

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/progan/progan3.png)

### Dataset Generation

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/progan/progan4.png)

## StyleGAN

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan/stylegan1.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan/stylegan2.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan/stylegan3.png)

## StyleGAN 2

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2/stylegan2-1.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2/stylegan2-2.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2/stylegan2-3.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2/stylegan2-4.png)

## StyleGAN 2 ADA

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2-ada/stylegan2-ada-1.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2-ada/stylegan2-ada-2.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan2-ada/stylegan2-ada-3.png)

## StyleGAN 3

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan3/stylegan3-1.png)

![alt text](https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gan/stylegan3/stylegan3-2.png)
