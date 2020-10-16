import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class ACGAN:
    def __init__(self):
        self.img_shape = (448, 320, 3)
        self.latent_shape = 100
        self.n_classes = 25

    def build_generator(self):
        base_height, base_width, base_channel = 7, 5, 3

        label = Input(shape=(1,))
        label_embed = Embedding(self.n_classes, 50)(label)
        label_embed = Dense(base_height * base_width * base_channel)(label_embed)
        label_embed = Reshape((base_height, base_width, base_channel))(label_embed)

        z = Input(shape=(self.latent_shape,))
        z_embedding = Dense((base_height * base_width * 125), activation='relu')(z)
        z_embedding = LeakyReLU()(z_embedding)
        z_embedding = Reshape((base_height, base_width, 125))(z_embedding)

        merge = Concatenate()([z_embedding, label_embed])

        gen = Conv2DTranspose(256, (11, 11), strides=(4, 4), padding="same")(merge)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        image_out = Conv2D(3, (7, 7), 1, activation='tanh', padding='same')(gen)

        return tf.keras.Model([z, label], image_out)

    def build_discriminator(self):
        label = Input(shape=(1,))
        label_embed = Embedding(self.n_classes, 50)(label)
        label_embed = Dense(self.img_shape[0] * self.img_shape[1] * self.img_shape[2])(label_embed)
        label_embed = Reshape((self.img_shape[0], self.img_shape[1], self.img_shape[2]))(label_embed)

        image = Input(shape=self.img_shape)
        merge = Concatenate()([image, label_embed])

        disc = Conv2D(64, (7, 7), strides=(4, 4), padding="same")(merge)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        disc = Conv2D(128, (5, 5), strides=(4, 4), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        disc = Conv2D(256, (5, 5), strides=(2, 2), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        flatten_disc = Flatten()(disc)
        flatten_disc = Dropout(0.5)(flatten_disc)

        valid = Dense(1, activation='sigmoid')(flatten_disc)

        return tf.keras.Model([image, label], valid)

    def build_acgan(self, generator, discriminator):
        z = Input(shape=(self.latent_shape,))
        label = Input(shape=(1,))

        img = generator([z, label])

        discriminator.trainable = False
        valid = discriminator([img, label])

        return tf.keras.Model([z, label], valid)
