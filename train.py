import numpy as np
import pickle
import tensorflow as tf
import model
import matplotlib.pyplot as plt


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
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.0002),
            metrics=['accuracy']
        )

        self.acgan = ACGAN.build_acgan(generator=self.generator, discriminator=self.discriminator)
        self.acgan.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.0002)
        )

        print("Loading dataset...")
        self.x = np.asarray(pickle.load(open("data/images.pkl", "rb")))
        self.y = np.asarray(pickle.load(open("data/genres.pkl", "rb")))

        self.x = (self.x.astype(np.float32) - 127.5) / 127.5  # normalize values for tanh activation function

        self.x = self.x.reshape(-1, 448, 320, 3)
        self.y = self.y.reshape(-1, 1)

    def generate_samples(self, epoch_no):
        r, c = 5, 5

        noise = self.sample_latent_noise(r * c)
        imgs, labels = self.sample_training_data(r * c)

        gen_imgs = self.generator.predict([noise, labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("training_samples/%d.png" % epoch_no)
        plt.close()

    def sample_training_data(self, batch_size):
        idx = np.random.randint(0, self.x.shape[0], batch_size)
        return self.x[idx], self.y[idx]

    def sample_latent_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.latent_shape)), np.random.randint(0, self.n_classes, batch_size)

    def train(self, sample_interval=100):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            real_imgs, real_labels = self.sample_training_data(self.batch_size)
            noise, fake_labels = self.sample_latent_noise(self.batch_size)

            gen_imgs = self.generator.predict([noise, fake_labels])

            d_loss_real = self.discriminator.train_on_batch([real_imgs, real_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, fake_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise, fake_labels = self.sample_latent_noise(self.batch_size)

            g_loss = self.acgan.train_on_batch([noise, fake_labels], valid)

            if epoch % 50 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.generate_samples(epoch)

        print("Training complete, saving model...")
        self.acgan.save("movie_ac_gan")


if __name__ == "__main__":
    Train = Train()
    Train.train()
