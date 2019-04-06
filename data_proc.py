import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

a = []
img_raw = tf.io.read_file('data/raw_images/0016E5_04620.png')
img_decoded_raw = tf.image.decode_image(img_raw,dtype=tf.float32)
a.append(img_decoded_raw)
plt.imshow(img_decoded_raw)
img_decoded_raw[0,0]
# image_size = (720,720)
# img_resized = tf.image.resize(img_decoded_raw,image_size)
# img_resized = img_resized / 255.0
#
# img_np = np.asarray(img_resized)
# plt.imshow(img_np)
# img_decoded_raw.shape
# img_labeled = tf.io.read_file('data/test_dir/0001TP_006690_L.png')
# img_decoded_labeled = tf.image.decode_image(img_labeled)
# img_decoded_labeled.shape
#
# img_d_resized /= 255.0
#
# plt.imshow(img_d_resized)
#
# img_d_resized[0,0]
# label_colors.index(list(np.asarray(img_d_resized[345,456],dtype='uint8')))




label_names = ['Animal',
                'Archway',
                'Bicyclist',
                'Bridge',
                'Building',
                'Car',
                'CartLuggagePram',
                'Child',
                'Column_Pole',
                'Fence',
                'LaneMkgsDriv',
                'LaneMkgsNonDriv',
                'Misc_Text',
                'MotorcycleScooter',
                'OtherMoving',
                'ParkingBlock',
                'Pedestrian',
                'Road',
                'RoadShoulder',
                'Sidewalk',
                'SignSymbol',
                'Sky',
                'SUVPickupTruck',
                'TrafficCone',
                'TrafficLight',
                'Train',
                'Tree',
                'Truck_Bus',
                'Tunnel',
                'VegetationMisc',
                'Void',
                'Wall']

label_colors = [[64, 128, 64],
                [192, 0, 128],
                [0, 128, 192],
                [0, 128, 64],
                [128, 0, 0],
                [64, 0, 128],
                [64, 0, 192],
                [192, 128, 64],
                [192, 192, 128],
                [64, 64, 128],
                [128, 0, 192],
                [192, 0, 64],
                [128, 128, 64],
                [192, 0, 192],
                [128, 64, 64],
                [64, 192, 128],
                [64, 64, 0],
                [128, 64, 128],
                [128, 128, 192],
                [0, 0, 192],
                [192, 128, 128],
                [128, 128, 128],
                [64, 128, 192],
                [0, 0, 64],
                [0, 64, 64],
                [192, 64, 128],
                [128, 128, 0],
                [192, 128, 192],
                [64, 0, 64],
                [192, 192, 0],
                [0, 0, 0],
                [64, 192, 0]]









def get_raw_images(raw_img_dir,num_images):
    '''
    function to get the input image tensor
    params:
    raw_img_dir : directory which contains all raw images
    returns:
    raw_images_tensor : shape = (num_images,height,width,3) containing all images
                        in raw_image_dir
    '''

    listdir = os.listdir(raw_img_dir)
    #num_images = len(listdir)
    img_raw = tf.io.read_file(raw_img_dir + "/" + listdir[1])
    img_decoded = tf.image.decode_image(img_raw)
    h = img_decoded.shape[0]
    w = img_decoded.shape[1]

    raw_images_tensor = np.zeros((num_images,
                                    h,
                                    w,
                                    3))

    for i in range(num_images):
        if(listdir[i].endswith('.png')):
            if(i > num_images):
                break
            img_raw = tf.io.read_file(raw_img_dir + "/" + listdir[i])
            img_decoded = tf.image.decode_image(img_raw)
            raw_images_tensor[i] = img_decoded

    return raw_images_tensor


def get_label_tensor(label_img_dir,num_images):
        '''
        function to get the label image tensor
        params:
        label_img_dir : directory which contains all label images
        returns:
        label_images_tensor : shape = (num_images,height,width,3) containing all images
                            in label_img_directory
        '''

        listdir = os.listdir(label_img_dir)
        #num_images = len(listdir)
        img_labeled = tf.io.read_file(label_img_dir + "/" + listdir[1])
        img_decoded = tf.image.decode_image(img_labeled)
        h = img_decoded.shape[0]
        w = img_decoded.shape[1]
        label_images_tensor = np.zeros((num_images,h,w,3))


        for i in range(num_images):
            if(listdir[i].endswith('.png')):
                if(i > num_images):
                    break
                print(label_img_dir + "/" + listdir[i])
                img_labeled = tf.io.read_file(label_img_dir + "/" + listdir[i])
                img_decoded = tf.image.decode_image(img_labeled)
                label_images_tensor[i] = img_decoded




        return label_images_tensor

def raw_images_generator(raw_img_dir):
    '''
    generator function to get the input image tensor
    params:
    raw_img_dir : directory which contains all raw images
    returns:
    raw_images_tensor : shape = (num_images,height,width,3) containing all images
                        in raw_image_dir
    '''

    listdir = os.listdir(raw_img_dir)

    for fname in listdir:
        if(fname.endswith('.png')):

            img_raw = tf.io.read_file(raw_img_dir + "/" + fname)
            img_decoded = tf.image.decode_image(img_raw,dtype=tf.float32)
            yield img_decoded

def label_images_generator(label_img_dir):

    listdir = os.listdir(label_img_dir)

    for fname in listdir:
        img_labeled = tf.io.read_file(label_img_dir + "/" + fname)
        img_decoded = tf.image.decode_image(img_raw,dtype=tf.float32)
        yield img_decoded

def training_generator(raw_img_dir,label_img_dir,batch_size):


    listdir = os.listdir(raw_img_dir)
    num_batches = int(len(listdir)/batch_size)

    for batch_num in range(num_batches):
        file_list = listdir[batch_num*batch_size:batch_num*batch_size+batch_size]
        size = len([fname for fname in file_list if fname.endswith('.png')])
        x_tensor = np.zeros((size,720,960,3))
        y_tensor = np.zeros((size,720,960,3))
        for i,fname in enumerate(file_list):
            if(fname.endswith('.png')):
                img_raw = tf.io.read_file(raw_img_dir + "/" + fname)
                img_labeled = tf.io.read_file(label_img_dir + "/" + fname[:-4] + "_L.png")
                img_decoded_raw = tf.image.decode_image(img_raw,dtype=tf.float32)
                img_decoded_labeled = tf.image.decode_image(img_labeled,dtype=tf.float32)
                x_tensor[i] = img_decoded_raw
                y_tensor[i] = img_decoded_labeled

        yield x_tensor, y_tensor



# Y = get_label_tensor('data/test_dir')
# X = get_raw_images('data/raw_images',10)
