from __future__ import print_function, division
from sklearn import datasets
import math
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import pdb

from sklearn.datasets import fetch_mldata

from optimizers import Adam
from loss_functions import SquareLoss
from layers import Dense, Activation, Reshape, BatchNormalization
from neural_network import NeuralNetwork

class Autoencoder():
    """An Autoencoder with deep fully-connected neural nets.

    Training Data: MNIST Handwritten Digits (28x28 images)
    """
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_dim = self.img_rows * self.img_cols
        self.latent_dim = 32 # The dimension of the data embedding

        optimizer = Adam(learning_rate=0.0002, b1=0.5)
        loss_function = SquareLoss

        self.encoder = self.build_encoder(optimizer, loss_function)
        self.decoder = self.build_decoder(optimizer, loss_function)

        self.autoencoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        self.autoencoder.layers.extend(self.encoder.layers)
        self.autoencoder.layers.extend(self.decoder.layers)
        self.autoencoder.back_opt = False

        print ()
        self.autoencoder.summary(name="Variational Autoencoder")

    def build_encoder(self, optimizer, loss_function):

        encoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        encoder.add(Dense(512, input_shape=(self.img_dim,)))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(256))
        encoder.add(Activation('leaky_relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.latent_dim))

        return encoder

    def build_decoder(self, optimizer, loss_function):

        decoder = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        decoder.add(Dense(256, input_shape=(self.latent_dim,)))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(512))
        decoder.add(Activation('leaky_relu'))
        decoder.add(BatchNormalization(momentum=0.8))
        decoder.add(Dense(self.img_dim))
        decoder.add(Activation('tanh'))

        return decoder

    def train(self, n_epochs, batch_size=128, save_interval=50):

        mnist = fetch_mldata('MNIST original')

        X = mnist.data

        # Split into training and test data
        np.random.shuffle(X)
        X, testX = X[:60000,:], X[60000:,:]

        # Rescale [-1, 1]
        X = (X.astype(np.float32) - 127.5) / 127.5
        testX = (testX.astype(np.float32) - 127.5) / 127.5

        for epoch in range(n_epochs):

            # Select a random half batch of images
            idx = np.random.randint(0, X.shape[0], batch_size)
            imgs = X[idx]

            # Train the Autoencoder
            loss, _ = self.autoencoder.train_on_batch(imgs, imgs)

            # Display the progress
            print ("%d [D loss: %f]" % (epoch, loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, testX)

    def save_imgs(self, epoch, X):
        r, c = 5, 5 # Grid size
        # Select a random half batch of images
        idx = np.random.randint(0, X.shape[0], r*c)
        imgs = X[idx]
        # Generate images and reshape to image shape
        gen_imgs = self.autoencoder.predict(imgs).reshape((-1, self.img_rows, self.img_cols))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        plt.suptitle("Autoencoder")
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("ae_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    ae = Autoencoder()
    ae.train(n_epochs=200000, batch_size=1, save_interval=40)