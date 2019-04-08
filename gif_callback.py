import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
This callback module can only be used for the following cases:
* If using keras.models.Model.fit()
* If using keras.models.Model.fit_generator()
'''

gif1_source = 'data/biomed/test/15.png'
gif2_source = 'data/biomed/test/2.png'
gif3_source = 'data/biomed/test/20.png'


img1 = tf.io.read_file(gif1_source)
img1 = tf.image.decode_image(img1,dtype=tf.float32)
img1 = tf.image.resize(img1,[256,256])
x1 = tf.expand_dims(img1,0)

img2 = tf.io.read_file(gif2_source)
img2 = tf.image.decode_image(img2,dtype=tf.float32)
img2 = tf.image.resize(img2,[256,256])
x2 = tf.expand_dims(img2,0)

img3 = tf.io.read_file(gif2_source)
img3 = tf.image.decode_image(img3,dtype=tf.float32)
img3 = tf.image.resize(img3,[256,256])
x3 = tf.expand_dims(img3,0)



class GIF_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        return

    def on_train_batch_begin(self,batch,logs={}):
        return

    def on_train_batch_end(self,batch,logs={}):
        return

    def on_epoch_end(self,epoch,logs={}):

        y_pred_1 = self.model(x1)
        plt.imsave('gif/1'+str(epoch)+'.png',np.asarray(y_pred_1[0,:,:,0]),cmap=plt.cm.gray,vmin=0.0,vmax=1.0)

        y_pred_2 = self.model(x2)
        plt.imsave('gif/2'+str(epoch)+'.png',np.asarray(y_pred_2[0,:,:,0]),cmap=plt.cm.gray,vmin=0.0,vmax=1.0)

        y_pred_3 = self.model(x3)
        plt.imsave('gif/3'+str(epoch)+'.png',np.asarray(y_pred_3[0,:,:,0]),cmap=plt.cm.gray,vmin=0.0,vmax=1.0)

        return
