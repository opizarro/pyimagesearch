from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Input, Flatten, Embedding, multiply, Dropout
from keras.layers import Concatenate, GaussianNoise,Activation
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras import backend as K

from pyimagesearch.preprocessing import ImageToArrayPreprocessor


import matplotlib as mpl
mpl.use('Agg') # does use DISPLAY
import matplotlib.pyplot as plt

import numpy as np

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.bp_rows = 21
        self.bp_cols = 21
        self.bp_channels = 1
        self.bp_shape = (self.bp_rows, self.bp_cols, self.bp_channels)
        self.latent_dim = 100

        #optimizer = Adam(0.0002, 0.5)
        optimizer = Adam(0.0001, 0.5)
        optimizer = Adam(0.0001, 0.5)
        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])


        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        bathy = Input(shape=(self.bp_shape))
        bathy_mean = Input(shape=(1,))

        img = self.generator([noise, bathy, bathy_mean])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, bathy, bathy_mean])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, bathy, bathy_mean], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        # latent space dimension
        z = Input(shape=(self.latent_dim,))

        # bathy
        bpatch = Input(shape=(self.bp_shape))
        bvec = Flatten()(bpatch)

        bathy_mean = Input(shape=(1,))

        # Generator network
        merged_layer = Concatenate()([z, bvec, bathy_mean])

        # FC: 2x2x512
        generator = Dense(2*2*512, activation='relu')(merged_layer)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = LeakyReLU(alpha=0.1)(generator)
        generator = Reshape((2, 2, 512))(generator)

        # # Conv 1: 4x4x256
        generator = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = LeakyReLU(alpha=0.1)(generator)

        # Conv 2: 8x8x128
        generator = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = LeakyReLU(alpha=0.1)(generator)

        # Conv 3: 16x16x64
        generator = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = LeakyReLU(alpha=0.1)(generator)


        # Conv 4: 32x32x32
        generator = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = LeakyReLU(alpha=0.1)(generator)

        # Conv 5: 64x64x3
        generator = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='linear')(generator)

        # generator = Model(inputs=[z, labels], outputs=out_g)
        gener = Model(inputs=[z, bpatch, bathy_mean], outputs=generator, name='generator')

        return gener

    def build_discriminator(self):

        # input image
        img_input = Input(shape=(self.img_shape))

        # bathy conditioning
        bpatch = Input(shape=(self.bp_shape))
        bvec = Flatten()(bpatch)

        bathy_mean = Input(shape=(1,))

        # Conv 1: 16x16x64
        discriminator = Conv2D(64, kernel_size=5, strides=2, padding='same')(img_input)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        discriminator = LeakyReLU(alpha=0.1)(discriminator)

        # Conv 2:
        discriminator = Conv2D(128, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        discriminator = LeakyReLU(alpha=0.1)(discriminator)

        # Conv 3:
        discriminator = Conv2D(256, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        discriminator = LeakyReLU(alpha=0.1)(discriminator)

        # Conv 4:
        discriminator = Conv2D(512, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        discriminator = LeakyReLU(alpha=0.1)(discriminator)

        # FC
        discriminator = Flatten()(discriminator)

        # Concatenate
        merged_layer = Concatenate()([discriminator, bvec])
        discriminator = Dense(512, activation='relu')(merged_layer)

        # Output
        discriminator = Dense(1, activation='sigmoid')(discriminator)

        discr = Model(inputs=[img_input, bpatch, bathy_mean], outputs=discriminator, name='discriminator')

        return discr

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset

        cached_images = '/data/bathy_training/cache_images_ohara_07.npz'
        # cached bathymetry
        cached_bpatches = '/data/bathy_training/cache_raw_bpatches_ohara_07.npz'
        all_bpatches = '/data/bathy_training/all_raw_bpatches_ohara_07.npz'
        # load dataset
        data =  np.load(cached_images)
        Ximg_train = data['xtrain']
        data = np.load(cached_bpatches)
        Xbathy_train = data['xtrain']
        Xbathy_train_means = np.mean(Xbathy_train,axis=(1,2))
        print("shape Xbathy_train_means ", Xbathy_train_means.shape)

        for k in np.arange(Xbathy_train.shape[0]):
            Xbathy_train[k,:,:,0] = Xbathy_train[k,:,:,0] - Xbathy_train_means[k]

        data = np.load(all_bpatches)
        Xbathy_all = data['xtrain']
        Xbathy_all_means = np.mean(Xbathy_all,axis=(1,2))
        for k in np.arange(Xbathy_all.shape[0]):
            Xbathy_all[k,:,:,0] = Xbathy_all[k,:,:,0] - Xbathy_all_means[k]

        # Configure input
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
#        X_train = np.expand_dims(X_train, axis=3)
#        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # to plot metrics
        d_loss_hist = []
        g_loss_hist = []

        # data generator
        dataGenerator = ImageDataGenerator(horizontal_flip = True, vertical_flip = True)
        batchIterator = dataGenerator.flow((Ximg_train,[Xbathy_train,Xbathy_train_means]), batch_size=batch_size)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            #idx = np.random.randint(0, Ximg_train.shape[0], batch_size)
            #imgs, bpatches, bp_means = Ximg_train[idx], Xbathy_train[idx], Xbathy_train_means[idx]

            imgs, bpatches, bp_means = batchIterator.next()
            #print("shapes inputs {}, {}, {}".format(imgs.shape,bpatches.shape,bp_means.shape))
            actual_batch_size =  imgs.shape[0]
            # Sample noise as generator input
            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise[:actual_batch_size], bpatches, bp_means])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, bpatches, bp_means], valid[:actual_batch_size])
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, bpatches, bp_means], fake[:actual_batch_size])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            # this is selecting random patches from a box around dive, presumably representative bathy

            idx = np.random.randint(0, Xbathy_all.shape[0], batch_size)
            random_bathy = Xbathy_all[idx]
            random_bathy_means = Xbathy_all_means[idx]
            #sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, random_bathy, random_bathy_means], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            d_loss_hist.append(d_loss[0])
            g_loss_hist.append(g_loss)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, Xbathy_train, Xbathy_train_means)

                # plotting the metrics
                plt.plot(d_loss_hist)
                plt.plot(g_loss_hist)
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Discriminator', 'Adversarial'], loc='best')
                plt.show()
                plt.savefig("metrics_cgan/metrics.png")
                plt.close()

    def sample_images(self, epoch, Xbathy_samples, Xbathy_samples_means):
        r, c = 2, 3
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
        idx = np.random.randint(0, Xbathy_samples.shape[0], r*c)
        random_bathy = Xbathy_samples[idx]
        random_bathy_means = Xbathy_samples_means[idx]

        #print("random bathy size ", random_bathy.shape)

        #sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, random_bathy, random_bathy_means])

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5
        print("img size ",gen_imgs.shape )
        fig, axs = plt.subplots(r, 2*c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(gen_imgs[cnt,::])
                #print("random bathy subset  size ", random_bathy[cnt,:,:,0].shape)
                axs[i,2*j+1].imshow(random_bathy[cnt,:,:,0])

                #axs[i,2*j+1].set_title("d {depth:.1f}".format(depth=random_bathy_means[cnt]))
                axs[i,2*j+1].set_title("d %.1f" % (random_bathy_means[cnt]))
                axs[i,2*j].axis('off')
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("images_cgan/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100000, batch_size=32, sample_interval=200)
