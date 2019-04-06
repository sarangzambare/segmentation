from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io


def prepare_data(img,label,is_multi_class,num_class):
    if(is_multi_class):
        img = img / 255
        label = label[:,:,:,0] if(len(label.shape) == 4) else label[:,:,0]
        new_label = np.zeros(label.shape + (num_class,))
        for i in range(num_class):
            new_label[label == i,i] = 1
        new_label = np.reshape(new_label,(new_label.shape[0],new_label.shape[1]*new_label.shape[2],new_label.shape[3])) if is_multi_class else np.reshape(new_label,(new_label.shape[0]*new_label.shape[1],new_label.shape[2]))
        label = new_label
    elif(np.max(img) > 1):
        img = img / 255
        label = label /255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (img,label)



def get_train_generator(batch_size,
                        train_dir,
                        image_path,
                        label_path,
                        aug_dict,
                        color_mode = "grayscale",
                        label_color_mode = "grayscale",
                        image_save_prefix  = "image",
                        label_save_prefix  = "label",
                        is_multi_class = False,
                        num_class = 2,
                        save_to_dir = None,
                        target_size = (256,256),
                        seed = 666):
    '''
    Generator function to generate image,label pairs, with transformations.
    params:
        batch_size: number of images per batch
        train_dir: directory containing training data
        image_path: directory in train_dir containing images
        label_path: directory in train_dir containing labels
        aug_dict: dictionery of augmentation parameters
        color_mode: grayscale
        label_color_mode: grayscale
        image_save_prefix: prefix for saved files
        label_save_prefix: prefix for saved files
        is_multi_class: if more than 2 classes or not
        num_class: number of classes
        save_to_dir: if not None, saves files to this directory
        target_size: target image size
        seed: random seed, must be same for train and label generators

    returns:
        image,label: pair of image and its corresponding label image

    '''
    image_gen = ImageDataGenerator(**aug_dict)
    label_gen = ImageDataGenerator(**aug_dict)
    image_generator = image_gen.flow_from_directory(
        train_dir,
        classes = [image_path],
        class_mode = None,
        color_mode = color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    label_generator = label_gen.flow_from_directory(
        train_dir,
        classes = [label_path],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = label_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, label_generator)
    for (img,label) in train_generator:
        img,label = prepare_data(img,label,is_multi_class,num_class)
        yield (img,label)


def test_generator(test_dir,num_images = 30,target_size = (256,256),is_multi_class = False,as_gray = True):
    for i in range(num_images):
        img = io.imread(os.path.join(test_dir,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not is_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
