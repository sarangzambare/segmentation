import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense
from tqdm import tqdm
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt


INPUT_SHAPE = (720,960,3)
OUTPUT_SHAPE = (720,960,3)
raw_dir = 'data/raw_images'
label_dir = 'data/labeled_images'

# def training_generator(raw_img_dir,label_img_dir):
#
#     listdir = os.listdir(raw_img_dir)
#     for fname in listdir:
#         if(fname.endswith('.png')):
#             img_raw = tf.io.read_file(raw_img_dir + "/" + fname)
#             img_labeled = tf.io.read_file(label_img_dir + "/" + fname[:-4] + "_L.png")
#             img_decoded_raw = tf.image.decode_image(img_raw,dtype=tf.float32)
#             img_decoded_labeled = tf.image.decode_image(img_labeled,dtype=tf.float32)
#
#             yield img_decoded_raw, img_decoded_labeled

class CONFIG:
    IMAGE_WIDTH = 960
    IMAGE_HEIGHT = 720
    COLOR_CHANNELS = 3




base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=INPUT_SHAPE)
base_model.summary()
feature_extractor = tf.keras.Model(inputs=base_model.layers[0].input,outputs=base_model.layers[15].output)
feature_extractor.trainable = False
img_raw = tf.io.read_file('data/raw_images/0016E5_04620.png')
img_decoded_raw = tf.image.decode_image(img_raw,dtype=tf.float32)
x = tf.expand_dims(img_decoded_raw,0)

features = feature_extractor(tf.expand_dims(img_decoded_raw,0))

plt.imshow(features[0,:,:,345])

def segmentator_model(input_shape):

    x_input = Input(input_shape)

    features = feature_extractor(x_input)

    X = Conv2DTranspose(filters=512,kernel_size=(3,4),padding='same',strides=(2,2),activation='relu')(features)
    X = Conv2DTranspose(filters=256,kernel_size=(3,4),padding='same',strides=(2,2),activation='relu')(X)
    X = Conv2DTranspose(filters=64,kernel_size=(3,4),padding='same',strides=(2,2),activation='relu')(X)
    X = Conv2D(filters=3,kernel_size=(1,1),padding='valid',activation='sigmoid')(X)


    return tf.keras.Model(inputs=x_input,outputs=X)






seg = segmentator_model(INPUT_SHAPE)

seg.summary()



def training_generator(raw_img_dir,label_img_dir,batch_size):


    listdir = os.listdir(raw_img_dir)
    num_batches = int(len(listdir)/batch_size)

    for batch_num in range(num_batches):
        file_list = listdir[batch_num*batch_size:batch_num*batch_size+batch_size]
        size = len([fname for fname in file_list if fname.endswith('.png')])
        x_tensor = np.zeros((size,720,960,3),dtype='float32')
        y_tensor = np.zeros((size,720,960,3),dtype='float32')
        for i,fname in enumerate(file_list):
            if(fname.endswith('.png')):
                img_raw = tf.io.read_file(raw_img_dir + "/" + fname)
                img_labeled = tf.io.read_file(label_img_dir + "/" + fname[:-4] + "_L.png")
                img_decoded_raw = tf.image.decode_image(img_raw,dtype=tf.float32)
                img_decoded_labeled = tf.image.decode_image(img_labeled,dtype=tf.float32)
                x_tensor[i] = img_decoded_raw
                y_tensor[i] = img_decoded_labeled

        yield x_tensor, y_tensor


# model = tf.keras.models.Sequential()
#
# model.add(InputLayer(input_shape = INPUT_SHAPE))
# model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=(2,2),activation='relu'))
# model.add(Conv2D(filters=192,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu'))
# model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(2,2),activation='relu'))
# model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu'))
# model.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding='same',strides=(2,2),activation='relu'))
# model.add(Conv2DTranspose(filters=192,kernel_size=(3,3),padding='same',strides=(1,1),activation='relu'))
# model.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding='same',strides=(2,2),activation='relu'))
# model.add(Conv2D(filters=3,kernel_size=(1,1),activation='relu'))
#model.add(Conv2D(filters=256,kernel_size=(3,4),padding='same',activation='relu'))
#model.add(Conv2DTranspose(filters=128,kernel_size=(3,4),padding='valid',strides=(2,2)))

def custom_loss(model,x,y):
    #assert(model.output.shape == OUTPUT_SHAPE)

    y_pred = model(x)

    loss = tf.reduce_mean(tf.square(tf.subtract(y,y_pred)))
    print("loss is :" + str(loss))
    return loss


def compute_gradients(model,x,y):
    with tf.GradientTape() as tape:
        loss = custom_loss(model,x,y)
        gradients = tape.gradient(loss,model.trainable_variables)

    return gradients, loss

def apply_gradients(optimizer,gradients,variables):
    optimizer.apply_gradients(zip(gradients,variables))



epochs = 1
optimizer = tf.keras.optimizers.Adam(1e-4)

for epoch in range(1,epochs+1):
    for x,y in training_generator(raw_dir,label_dir,1):

        print("reference  string: " + str(x[0,0,0]) + " " +str(y[0,0,0]))
        gradients, loss = compute_gradients(seg,x,y)
        #print('gradients: ' + str(gradients))
        apply_gradients(optimizer,gradients,seg.trainable_variables)
