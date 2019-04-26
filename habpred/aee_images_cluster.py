from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Cropping2D
from keras.layers import Concatenate, MaxPooling2D, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib as mpl
mpl.use('Agg') # does use DISPLAY
import matplotlib.pyplot as plt
# set small title size
plt.rcParams.update({'axes.titlesize':'xx-small'})

import numpy as np
# added by OP
from scipy.stats import bernoulli
from keras.layers import Lambda
from keras import regularizers
from benthic_utils import bathy_data, benthic_img_data
from keras.utils import multi_gpu_model
import tensorflow as tf


class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64
        self.latent_catdim = 5

        optimizerD = Adam(lr=1e-6, decay=1e-6)
        optimizerA = Adam(lr=1e-4, decay=1e-6)
        # Build and compile the discriminator
        #self.discriminator = self.build_discriminator()
        with tf.device('/gpu:0'):
            self.discriminator = self.model_discriminator()
            self.discriminator_cat = self.model_discriminator_cat()
        # try using multi_gpu
        #try:
        #self.discriminator = multi_gpu_model(self.discriminator, cpu_relocation=True)
        #    print("Training discriminator using multiple GPUs ...")
        #except:
        #    print("Training discriminator on singe GPU or CPU")


        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizerD,
            metrics=['accuracy'])
        self.discriminator_cat.compile(loss='binary_crossentropy',
            optimizer=optimizerD,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.model_encoder()
        self.decoder = self.model_generator()

        # inputs to encoder
        img = Input(shape=self.img_shape)

        # The autoencoder takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr, latent_cat = self.encoder(img)
        reconstructed_img = self.decoder([encoded_repr,latent_cat])

        # The discriminator determines validity of the encoding

        validity = self.discriminator(encoded_repr)
        validity_cat = self.discriminator_cat(latent_cat)
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator_cat.trainable = False
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        with tf.device('/gpu:0'):
            self.adversarial_autoencoder = Model([img], [reconstructed_img, validity, validity_cat])

        # try using multi_gpu
        #try:
        #    self.adversarial_autoencoder = multi_gpu_model(self.adversarial_autoencoder, cpu_relocation=True)
        #    print("Training autoencoder using multiple GPUs ...")
        #except:
        #    print("Training autoencoder on singe GPU or CPU")

        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[100, 1e-2, 1e-2],
            optimizer=optimizerA)
    #    print("Autoencoder metrics {}".format(self.adversarial_autoencoder.metrics_names))

    def model_encoder(self, units=512, reg=lambda: regularizers.l1_l2(l1=1e-7, l2=1e-7), dropout=0.5):
# image and bathy encoder assumes 64 x 64 image, 21 x 21 bathy + one mean depth
        k = 5
        x = Input(shape=self.img_shape)
        h = Conv2D(units// 16, (k, k), activation='linear', use_bias=True, padding='same', kernel_regularizer=reg())(x)
        #h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 32 x 32
        h = PReLU()(h)
        h = Conv2D(units // 8, (k, k), activation='linear',use_bias=True,  padding='same', kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 16 x 16
        h = PReLU()(h)
        h = Conv2D(units // 4, (k, k), activation='linear', use_bias=True, padding='same', kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 8 x 8
        h = PReLU()(h)
        h = Conv2D(units // 2, (k, k), activation='linear', use_bias=True, padding='same', kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 2))(h) # 4 x 4
        h = PReLU()(h)
        h = Conv2D(units, (k, k), activation='linear', use_bias=True, padding='same', kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = PReLU()(h)
        h = Flatten()(h)
        mu = Dense(self.latent_dim, name="encoder_mu", kernel_regularizer=reg())(h)
        log_sigma_sq = Dense(self.latent_dim, name="encoder_log_sigma_sq", kernel_regularizer=reg())(h)
        # z = Lambda(lambda (_mu, _lss): _mu + K.random_normal(K.shape(_mu)) * K.exp(_lss / 2),output_shape=lambda (_mu, _lss): _mu)([mu, log_sigma_sq])
        z = Lambda(lambda ml: ml[0] + K.random_normal(K.shape(ml[0])) * K.exp(ml[1] / 2),
                   output_shape=lambda ml: ml[0])([mu, log_sigma_sq])
        y = Dense(self.latent_catdim, name="encoder_categories", use_bias=True, activation="softmax", kernel_regularizer=reg())(h)


        return Model(x, [z,y], name="encoder")


    def model_generator(self, units=512):

        # latent space dimension
        z = Input(shape=(self.latent_dim,))
        y = Input(shape=(self.latent_catdim,))

        # FC: preprocess latent inage data
        zgenerator = Dense(units//4, activation='relu')(z)
        zgenerator = BatchNormalization(momentum=0.9)(zgenerator)
        zgenerator = LeakyReLU(alpha=0.1)(zgenerator)

        zgenerator = Dense(units//4, activation='relu')(z)
        zgenerator = BatchNormalization(momentum=0.9)(zgenerator)
        zgenerator = LeakyReLU(alpha=0.1)(zgenerator)


        # FC: preprocess categorical data
        ygenerator = Dense(units//4, activation='relu')(y)
        ygenerator = BatchNormalization(momentum=0.9)(ygenerator)
        ygenerator = LeakyReLU(alpha=0.1)(ygenerator)

        ygenerator = Dense(units//4, activation='relu')(y)
        ygenerator = BatchNormalization(momentum=0.9)(ygenerator)
        ygenerator = LeakyReLU(alpha=0.1)(ygenerator)


        # Generator network
        merged_layer = Concatenate()([zgenerator, ygenerator])

        # FC: 2x2x512
        generator = Dense(2 * 2 * units, activation='relu')(merged_layer)
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
        generator = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(generator)


        # generator = Model(inputs=[z, labels], outputs=out_g)
        gener = Model(inputs=[z, y], outputs=generator, name='generator')
        gener.summary()
        return gener






    def model_discriminator(self, output_dim=1, units=512, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
        z = Input(shape=(self.latent_dim,))
        h = z
        h = Dense(units, name="discriminator_h1", use_bias=True, kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h2", use_bias=True, kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h3", use_bias=True, kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        h = PReLU()(h)
        y = Dense(output_dim, name="discriminator_y", use_bias=True, activation="sigmoid", kernel_regularizer=reg())(h)
        #h = BatchNormalization()(h)
        #y = Activation('sigmoid')(h)
        return Model(z, y)

    def model_discriminator_cat(self, output_dim=1, units=128, reg=lambda: regularizers.l1_l2(1e-7, 1e-7)):
        z = Input(shape=(self.latent_catdim,))
        h = z
        h = Dense(units, name="discriminator_h1", use_bias=True, kernel_regularizer=reg())(h)
        # h = BatchNormalization()(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h2", use_bias=True, kernel_regularizer=reg())(h)
        # h = BatchNormalization()(h)
        h = PReLU()(h)
        h = Dense(units // 2, name="discriminator_h3", use_bias=True, kernel_regularizer=reg())(h)
        # h = BatchNormalization()(h)
        h = PReLU()(h)
        y = Dense(output_dim, name="discriminator_y", use_bias=True, activation="sigmoid", kernel_regularizer=reg())(h)
        # h = BatchNormalization()(h)
        # y = Activation('sigmoid')(h)
        return Model(z, y)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        (Ximg_train,_) = benthic_img_data()
        print("shape Ximg_train {}".format(Ximg_train.shape))

        print(Ximg_train.shape[0], 'train img samples')

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3) # only used if image is 2D without channel info

        desc_batch = int(batch_size / 2)
        #desc_batch = int(batch_size)

        #noise_frac = 0.05
        #missing_prob = 0.5

        # plotting metrics
        d_loss_hist = []
        g_loss_hist = []

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, Ximg_train.shape[0], desc_batch)
            imgs = Ximg_train[idx]


            #print("shape imgs {}".format(imgs.shape))
            latent_fake, latent_fake_cat = self.encoder.predict([imgs])
            latent_real = np.random.normal(size=(desc_batch, self.latent_dim))
                # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, np.ones((desc_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, np.zeros((desc_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            sampled_labels = np.random.randint(0, self.latent_catdim, desc_batch).reshape(-1, 1)
            latent_real_cat = to_categorical(sampled_labels, num_classes=self.latent_catdim)

            # Train the discriminator cat
            d_loss_real_cat = self.discriminator_cat.train_on_batch(latent_real_cat, np.ones((desc_batch, 1)))
            d_loss_fake_cat = self.discriminator_cat.train_on_batch(latent_fake_cat, np.zeros((desc_batch, 1)))
            d_loss_cat = 0.5 * np.add(d_loss_real_cat, d_loss_fake_cat)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, Ximg_train.shape[0], batch_size)
            imgs = Ximg_train[idx]

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, np.ones((batch_size, 1)), np.ones((batch_size, 1))])
            # Train the generator

            # Plot the progress
            print ("imgs %d [D loss: %f, acc: %.2f%%] [Dcat loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], d_loss_cat[0], 100*dloss_cat[1], g_loss[0], g_loss[1]))


            d_loss_hist.append(d_loss*100)
            g_loss_hist.append((g_loss[0],g_loss[1]*1000))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.sample_autoencoder(epoch, imgs, save_folder="cat_images_aae")

                    # plotting the metrics
                plt.plot(d_loss_hist,linewidth=0.5)
                plt.plot(g_loss_hist,linewidth=0.5)
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Dloss*100', 'Dacc*100','AEloss','AEmse*1k'], loc='center right')
                plt.show()
                plt.savefig("cat_metrics_aae_img/aae_img_metrics.png")
                plt.close()
        #save params once done training
        self.save_model()
        self.save_latentspace(Ximg_train,"z_img_cat")

    def sample_images(self, epoch):
        r = 4
        c = self.latent_catdim
        fig, axs = plt.subplots(r, 2*c)
        cnt = 0
        for i in range(r):
            z = np.random.normal(size=(1,self.latent_dim))
            for j in range(c):
                cat = np.zeros((1,c))
                cat[0,j] = 1
                print("col {} cat {}".format(j,cat))
                gen_imgs = self.decoder.predict([z,cat])
                axs[i,2*j].imshow(gen_imgs[0])
                axs[i,2*j].axis('off')
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig("cat_images_aae_generator/benthic_%d.png" % epoch)
        plt.close()

    def sample_autoencoder(self, epoch,imgs, save_folder):
        r, c = 4, 2
        namps = r*c

        # Select a random set of images
        #idx = np.random.randint(0, X_train.shape[0], nsamps)
        #imgs = X_train[idx]
        gen_imgs, valids, valids_cat = self.adversarial_autoencoder.predict(imgs)

        #gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c*2)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,2*j].imshow(imgs[cnt])
                axs[i,2*j].axis('off')
                axs[i,2*j+1].imshow(gen_imgs[cnt])
                axs[i,2*j+1].axis('off')
                cnt += 1
        fig.savefig(save_folder+"/benthic_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.encoder, "img_aae_encoder_cat")
        save(self.decoder, "img_aae_decoder_cat")
        save(self.discriminator, "img_aae_discriminator_cat")


    def save_latentspace(self, inputdata, latent_name):

        z = self.encoder.predict(inputdata)
        np.save("saved_latent/%s.npy" % latent_name, z)


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=200000, batch_size=32, sample_interval=500)
