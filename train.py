import numpy as np
import pickle
import tensorflow as tf
import model


class Train:
    def __init__(self):
        self.epochs = 12000
        self.batch_size = 32

        self.img_shape = (448, 320, 3)
        self.latent_shape = 100
        self.n_classes = 25

        ACGAN = model.ACGAN()
        self.generator = ACGAN.build_generator()

        self.discriminator = ACGAN.build_discriminator()
        self.discriminator.compile(
            loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.SparseCategoricalCrossentropy()],
            optimizer=tf.keras.optimizers.Adam(0.0003),
            metrics=['accuracy']
        )

        self.acgan = ACGAN.build_acgan(generator=self.generator, discriminator=self.discriminator)
        self.acgan.compile(
            loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.SparseCategoricalCrossentropy()],
            optimizer=tf.keras.optimizers.Adam(0.0003)
        )

        print("Loading dataset...")
        self.x = np.asarray(pickle.load(open("data/images.pkl", "rb")))
        self.y = np.asarray(pickle.load(open("data/genres.pkl", "rb")))

        self.x = (self.x.astype(np.float32) - 127.5) / 127.5  # normalize values for tanh activation function

        self.x = self.x.reshape(-1, 448, 320, 3)
        self.y = self.y.reshape(-1, 1)

    def generate_samples(self, epoch_no):
        pass

    def sample_training_data(self, batch_size):
        idx = np.random.randint(0, self.x.shape[0], batch_size)
        return self.x[idx], self.y[idx]

    def sample_latent_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.latent_shape))

    def train(self, sample_interval=100):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            imgs, labels = self.sample_training_data(self.batch_size)
            noise = self.sample_latent_noise(self.batch_size)

            gen_imgs = self.generator.predict([noise, labels])

            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.acgan.train_on_batch([noise, labels], [valid, labels])

            if epoch % 50 == 0:
                print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            if epoch % sample_interval == 0:
                self.generate_samples(epoch)

        print("Training complete, saving model...")
        self.acgan.save("movie_ac_gan")


if __name__ == "__main__":
    Train = Train()
    Train.train()
