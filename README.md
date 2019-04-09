# Segmentation of neural structures in EM images, using tensorflow 2.0

Tensorflow 2.0 is here, session is gone. This repository uses tensorflow 2.0 to train a fully convolutional segmentation model, consisting of downsampling and upsampling convolutional layers.

*Semantic segmentation is the process of assigning each pixel in a given image a class, to get a better understanding of the image or for further processing.*

## The problem
In this case, we deal with a set of 30 consecutive images (512 × 512 pixels) from a serial section Transmission Electron Microscopy (ssTEM) dataset of the Drosophila first instar larva ventral nerve cord ([VNC; Cardona et al., 2010](https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full#B6)). The aim is to assign each pixel to either a "0" for belonging to the boundary between neurons, or "1" for belonging to the inside of the cell, resulting into a binary image with "black" for cell boundaries and "white" for other areas.

Boundary detection is challenging because many boundaries look fuzzy and ambiguous. Furthermore, only boundaries between neurites should be detected, and those of intracellular organelles like mitochondria and synaptic vesicles should be ignored.
