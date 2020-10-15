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
        label_embedding = Embedding(self.n_classes, 50)(label)
        label_embedding = Dense(base_height * base_width * base_channel)(label_embedding)
        label_embedding = Reshape((base_height, base_width, base_channel))(label_embedding)

        z = Input(shape=(self.latent_shape,))
        z_embedding = Dense((base_height * base_width * 381), activation='relu')(z)
        z_embedding = Reshape((base_height, base_width, 381))(z_embedding)

        merge = Concatenate()([z_embedding, label_embedding])

        gen = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding="same")(merge)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same")(gen)
        gen = BatchNormalization()(gen)
        gen = Activation("relu")(gen)

        gen = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same")(gen)

        image_out = Activation("tanh")(gen)

        return tf.keras.Model([z, label], image_out)

    def build_discriminator(self):
        image = Input(shape=self.img_shape)

        disc = Conv2D(32, (5, 5), strides=(3, 3), padding="same")(image)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        disc = Conv2D(64, (5, 5), strides=(2, 2), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        disc = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)

        disc = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)
        disc = Dropout(0.5)(disc)

        disc = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(disc)
        disc = BatchNormalization()(disc)
        disc = LeakyReLU()(disc)
        disc = Dropout(0.5)(disc)

        flatten_disc = Flatten()(disc)

        valid = Dense(1, activation='sigmoid')(flatten_disc)
        class_pred = Dense(self.n_classes, activation='softmax')(flatten_disc)

        return tf.keras.Model(image, [valid, class_pred])

    def build_acgan(self, generator, discriminator):
        z = Input(shape=(self.latent_shape,))
        label = Input(shape=(1,))

        img = generator([z, label])

        discriminator.trainable = False
        valid, class_pred = discriminator(img)

        return tf.keras.Model([z, label], [valid, class_pred])
