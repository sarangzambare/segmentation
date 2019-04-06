import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tqdm import tqdm
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt

INPUT_SHAPE = (256,256,1)
OUTPUT_SHAPE = (256,256,1)
train_dir = 'data/biomed/train/image/'
label_dir = 'data/biomed/train/label/'


img_raw = tf.io.read_file(train_dir+'0.png')
img_label = tf.io.read_file(label_dir+'0.png')
img_decoded_raw = tf.image.resize(tf.image.decode_image(img_raw,dtype=tf.float32),size=[256,256])
img_decoded_label = tf.image.resize(tf.image.decode_image(img_label,dtype=tf.float32),size=[256,256])
img_decoded_label[0,0]
x = tf.expand_dims(img_decoded_raw,0)
y = tf.expand_dims(img_decoded_label,0)






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


def training_generator(raw_img_dir,label_img_dir,batch_size):


    listdir = os.listdir(raw_img_dir)
    num_batches = int(len(listdir)/batch_size)

    for batch_num in range(num_batches):
        file_list = listdir[batch_num*batch_size:batch_num*batch_size+batch_size]
        size = len([fname for fname in file_list if fname.endswith('.png')])
        x_tensor = np.zeros((size,256,256,1),dtype='float32')
        y_tensor = np.zeros((size,256,256,1),dtype='float32')
        for i,fname in enumerate(file_list):
            if(fname.endswith('.png')):
                img_raw = tf.io.read_file(raw_img_dir + fname)
                img_labeled = tf.io.read_file(label_img_dir + fname)
                img_decoded_raw = tf.image.resize(tf.image.decode_image(img_raw,dtype=tf.float32),size=[256,256])
                img_decoded_labeled = tf.image.resize(tf.image.decode_image(img_labeled,dtype=tf.float32),size=[256,256])
                x_tensor[i] = img_decoded_raw
                y_tensor[i] = img_decoded_labeled

        yield x_tensor, y_tensor

def segmentator_model(input_shape):

    x_input = Input(input_shape)

    X = Conv2D(filters=64,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(x_input)
    X = Conv2D(filters=64,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X_SKIP_1 = X
    print('skip1 shape is :' + str(X_SKIP_1.shape))

    X = MaxPooling2D()(X)
    X = Conv2D(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X = Conv2D(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X_SKIP_2 = X
    print('skip2 shape is :' + str(X_SKIP_2.shape))

    X = Conv2DTranspose(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X = Conv2DTranspose(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X = UpSampling2D(size=(2,2))(X)
    X = concatenate([X_SKIP_1,X])

    X = Conv2DTranspose(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)
    X = Conv2DTranspose(filters=128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X)

    X = Conv2D(filters=1,kernel_size=(1,1),padding='valid',strides=(1,1),activation='sigmoid')(X)





    #X = Conv2D(filters=64,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu')(X_SKIP_1)




    return tf.keras.Model(inputs=x_input,outputs=X)





seg = segmentator_model(INPUT_SHAPE)
seg.summary()

epochs = 5
optimizer = tf.keras.optimizers.Adam(1e-3)

for epoch in range(1,epochs+1):
    print('Epoch no. ' + str(epoch))
    for x,y in training_generator(train_dir,label_dir,10):

        print("reference  string: " + str(x[0,0,0]) + " " +str(y[0,0,0]))
        gradients, loss = compute_gradients(seg,x,y)
        #print('gradients: ' + str(gradients))
        apply_gradients(optimizer,gradients,seg.trainable_variables)


y_pred = seg(x)


plt.imshow(x[0,:,:,0])
plt.imshow(np.asarray(y_pred[0,:,:,0]))
