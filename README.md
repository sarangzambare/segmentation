# Segmentation of neural structures in EM images, using tensorflow 2.0

## Author : Sarang Zambare

Tensorflow 2.0 is here, session is gone. This repository uses tensorflow 2.0 to train a convolutional segmentation model, consisting of downsampling and upsampling layers.

Input image             |  Segmented image, animated over epochs.
:-------------------------:|:-------------------------:
![alt-text](https://raw.githubusercontent.com/sarangzambare/segmentation/master/png/15copy.png)  |  ![alt-text](https://raw.githubusercontent.com/sarangzambare/segmentation/master/png/1.GIF)

*Semantic segmentation is the process of assigning each pixel in a given image a class, to get a better understanding of the image or for further processing.*

## The problem
In this case, we deal with a set of 30 consecutive images (512 × 512 pixels) from a serial section Transmission Electron Microscopy (ssTEM) dataset of the Drosophila first instar larva ventral nerve cord ([VNC; Cardona et al., 2010](https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full#B6)). **The aim is to assign each pixel to either a "0" for belonging to the boundary between neurons, or "1" for belonging to the inside of the cell, resulting into a binary image with "black" for cell boundaries and "white" for other areas.**

Boundary detection is challenging because many boundaries look fuzzy and ambiguous. Furthermore, only boundaries between neurites should be detected, and those of intracellular organelles like mitochondria and synaptic vesicles should be ignored.  

## Approach :

The idea is to have a neural network architecture consisting of:
1. Downsampling layers: which would be responsible for feature extraction from the image. These layers are the conventional convolutional layers, along with maxpooling layers to downsample the image. Think of this as projecting the image into a lower dimensional space.
2. Upsampling layers : These layers transform their inputs into higher dimensional representations. A way to do this for images is using transposed convolution, which can be thought of as a reversed way of doing the convolution operation (well not exactly). The difference between normal convolution and transposed convolution can be understood well by the gifs below by [Vincent Dumoulin, Francesco Visin](https://github.com/vdumoulin/conv_arithmetic) :


## Convolution animations
*Blue maps are inputs, and cyan maps are outputs*
<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="png/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="png/arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="png/padding_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Padding, strides</td>
  </tr>
</table>

## Transposed convolution animations (Upsampling)
*Blue maps are inputs, and cyan maps are outputs*
<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="png/no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="png/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="png/padding_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Padding, strides, transposed</td>
  </tr>
</table>

The base idea here is to design a dnn, which encodes the features using downsampling layers, and then generates segmented images using upsampling layers.

This is a supervised process, and for each input image, the label is grayscale image consisting of black cell boundaries and white interiors of the cell.

![alt-text](https://raw.githubusercontent.com/sarangzambare/segmentation/master/png/data.gif)



For this specific problem, a host of non-machine learning approaches might give better results as well. (e.g. Edge detection algorithms).
