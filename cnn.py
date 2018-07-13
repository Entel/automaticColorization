import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

HEIGHT = 255
WIDTH = 255
DEPTH = 3
BATCH_SIZE = 20

class CNN(object):
    def __init__(self):
        self.model = self.__model__()
        self.tensorboard = TensorBoard(log_dir = '../logs')
        self.X = []
        for filename in os.listdir('../colornet/'):
            self.X.append(img_to_array(load_img('colornet' + filename)))
        self.X = np.array(self.X, dtype = 'float')
        self.Xtrain = 1.0/255 * self.X
        self.datagen = ImageDataGenerator(
            shear_range = 0.2,
            zoom_range = 0.2,
            rotation_range = 20,
            horizontal_flip = True)

        self.inception = self.load_weight()

    def __model__(self):
        input_shape = (HEIGHT, WIDTH, DEPTH)
    
        embed_input = Input(shape=(1000,))

        #Encoder
        encoder_input = Input(shape=(HEIGHT, WIDTH, 1,))
        encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

        #Fusion
        fusion_output = RepeatVector(32 * 32)(embed_input) 
        fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
        fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
        fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

        #Decoder
        decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)

        model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
        return model

    def load_weight(self):
        inception = InceptionResNetV2(weights = None, include_top = True)
        inception.load_weights('../inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
        inception.graph = tf.get_default_graph()
        return inception  

    def create_inception_embedding(self, grayscaled_rgb):
        grayscaled_rgb_resized = []
        for i in grayscaled_rgb:
            i = resize(i, (HEIGHT, WIDTH, DEPTH), mode = 'constant')
            grayscaled_rgb_resized.append(i)
        grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
        grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
        with self.inception.graph.as_default():
            embed = inception.predict(grayscaled_rgb_resized)
        return embed

    def image_a_b_gen(self, batch_size):
        for batch in self.datagen.flow(self.Xtrain, batch_size = batch_size):
            grayscaled_rgb = gray2rgb(rgb2gray(batch))
            embed = create_inception_embedding(grayscaled_rgb)
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            X_batch = X_batch.reshape(X_batch.shape+(1,))
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

    def training(self):
        model = self.model
        model.compile(optimizer = 'adam', loss = 'mse')
        model.fit_generator(self.image_a_b_gen(BATCH_SIZE),
                callbacks = [self.tensorboard],
                epochs = 100,
                steps_per_epoch = 20)

        model_json = model.to_json()
        with open('../model/model.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weight('color_net.h5')

if __name__ == '__main__':
    cnn = CNN()
    cnn.training()
