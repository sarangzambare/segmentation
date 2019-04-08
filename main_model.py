import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tqdm import tqdm
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from data_pre import *




class CONFIG:

    INPUT_SHAPE = (256,256,1)
    OUTPUT_SHAPE = (256,256,1)
    train_dir = 'data/biomed/train/image/'
    label_dir = 'data/biomed/train/label/'



#Testing Cases:
# img_raw = tf.io.read_file(train_dir+'0.png')
# img_label = tf.io.read_file(label_dir+'0.png')
# img_decoded_raw = tf.image.resize(tf.image.decode_image(img_raw,dtype=tf.float32),size=[256,256])
# img_decoded_label = tf.image.resize(tf.image.decode_image(img_label,dtype=tf.float32),size=[256,256])
# img_decoded_label[0,0]
# x = tf.expand_dims(img_decoded_raw,0)
# y = tf.expand_dims(img_decoded_label,0)






def custom_loss(model,x,y):
    #assert(model.output.shape == OUTPUT_SHAPE)

    y_pred = model(x)

    loss = tf.reduce_mean(-y*tf.math.log(y_pred)-(1-y)*tf.math.log(1-y_pred))
    print("loss is :" + str(loss))
    return loss


def compute_gradients(model,x,y):
    with tf.GradientTape() as tape:
        loss = custom_loss(model,x,y)
        gradients = tape.gradient(loss,model.trainable_variables)

    return gradients, loss

def apply_gradients(optimizer,gradients,variables):
    optimizer.apply_gradients(zip(gradients,variables))




def segmentator_model(input_shape):

    x_inputs = Input(input_shape)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x_inputs)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x_skip_4 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x_skip_3 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x_skip_2 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Dropout(0.5)(x)
    x_skip_1 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(filters=512,kernel_size=2,activation='relu',padding='same',strides=2,kernel_initializer='he_normal')(x)
    x = concatenate([x_skip_1,x], axis = 3)
    x = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = Conv2DTranspose(filters=256,kernel_size=2,activation='relu',padding='same',strides=2,kernel_initializer='he_normal')(x)
    x = concatenate([x_skip_2,x], axis = 3)
    x = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = Conv2DTranspose(filters=256,kernel_size=2,activation='relu',padding='same',strides=2,kernel_initializer='he_normal')(x)
    x = concatenate([x_skip_3,x], axis = 3)
    x = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    x = Conv2DTranspose(filters=128,kernel_size=2,activation='relu',padding='same',strides=2,kernel_initializer='he_normal')(x)
    x = concatenate([x_skip_4,x], axis = 3)
    x = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(x)



    return tf.keras.Model(inputs=x_inputs,outputs=x)



data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_generator = get_train_generator(2,'data/biomed/train','image','label',data_gen_args,save_to_dir = None)

seg = segmentator_model(CONFIG.INPUT_SHAPE)
#seg.summary()

epochs = 10
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)
steps_per_epoch = 1000

for epoch in range(1,epochs+1):
    print('Epoch no. ' + str(epoch))
    for i,(x,y) in enumerate(train_generator):

        print("reference  string: " + str(x[0,0,0]) + " " +str(y[0,0,0]))
        gradients, loss = compute_gradients(seg,x,y)
        #print('gradients: ' + str(gradients))
        apply_gradients(optimizer,gradients,seg.trainable_variables)
        if(i > steps_per_epoch):
            break


seg.save('model_' + str(epochs) + 'epochs_' + str(steps_per_epoch) + 'spe_' + str(learning_rate) + 'lr.h5'  )
