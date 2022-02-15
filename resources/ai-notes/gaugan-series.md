# [AI Notes](https://github.com/quickgrid/AI-Resources/blob/master/ai-notes.md)

Notes from reading various deep learning, computer vision etc. papers. 

Many of the text are copied from paper verbatim, others with some modifications and rephrasing. Figures are from paper. Highlighted portions should be read and only some of the highlighted parts are expanded as notes.

**WARNING:** These notes may contain errors due to misinterpretation, lack of understanding, missing details etc. 

### TODO

- Fill in important highlighted missing details.

# GauGAN Family

### Papers

| Paper | Year | Conference |
| --- | --- | --- |
| [Semantic Image Synthesis with Spatially-Adaptive Normalization (GauGAN/SPADE)](https://arxiv.org/abs/1903.07291) | 2019 |  |

# GauGAN

## GauGAN Generation

<img src="figures/gaugan/gaugan_4.png" width=90% height=90%>

<img src="figures/gaugan/gaugan_10.png" width=90% height=90%>


## Overall Structure

<img src="figures/gaugan/gaugan_1.png" width=50% height=50%>

## Image Encoder

<img src="figures/gaugan/gaugan_2.png" width=50% height=50%>

## Generator

### Overall Structure

<img src="figures/gaugan/gaugan_5.png" width=50% height=50%>


### SPADE ResBlk (Residual Block) 

<img src="figures/gaugan/gaugan_6.png" width=50% height=50%>


### SPADE Block 

<img src="figures/gaugan/gaugan_7.png" width=50% height=50%>

## Discriminator

<img src="figures/gaugan/gaugan_3.png" width=50% height=50%>

## Additional Implementation Details

<img src="figures/gaugan/gaugan_8.png" width=50% height=50%>



## Spatially Adaptive Denormalization (SPADE)

<img src="figures/gaugan/gaugan_11.png" width=90% height=90%>

<img src="figures/gaugan/gaugan_12.png" width=50% height=50%>
