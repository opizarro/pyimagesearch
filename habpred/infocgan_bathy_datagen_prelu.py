from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, PReLU
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Input, Flatten, Embedding, multiply, Dropout
from keras.layers import Concatenate, GaussianNoise,Activation
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras import backend as K

import matplotlib as mpl
mpl.use('Agg') # does use DISPLAY
import matplotlib.pyplot as plt

import numpy as np

class INFOCGAN():
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
        self.latent_dim = 28
        self.num_classes = 5

        #optimizer = Adam(0.0002, 0.5)

        optimizer = Adam(0.0001, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]
        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the discriminator and recognition network
        self.discriminator, self.auxiliary = self.build_discriminator_and_q_net()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxiliary.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

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
        # The recognition network produces the label
        target_label = self.auxiliary([img, bathy, bathy_mean])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, bathy, bathy_mean], [valid, target_label])
        self.combined.compile(loss=losses,
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
        generator = Dense(2*2*512, activation='linear')(merged_layer)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = PReLU()(generator)
        generator = Reshape((2, 2, 512))(generator)

        # # Conv 1: 4x4x256
        generator = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = PReLU()(generator)

        # Conv 2: 8x8x128
        generator = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = PReLU()(generator)

        # Conv 3: 16x16x64
        generator = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = PReLU()(generator)


        # Conv 4: 32x32x32
        generator = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(generator)
        generator = BatchNormalization(momentum=0.9)(generator)
        generator = PReLU()(generator)

        # Conv 5: 64x64x3
        generator = Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')(generator)

        # generator = Model(inputs=[z, labels], outputs=out_g)
        gener = Model(inputs=[z, bpatch, bathy_mean], outputs=generator, name='generator')

        gener.summary()

        return gener

    def build_discriminator_and_q_net(self):

        # input image
        img_input = Input(shape=(self.img_shape))

        # bathy conditioning
        bpatch = Input(shape=(self.bp_shape))
        bvec = Flatten()(bpatch)

        bathy_mean = Input(shape=(1,))

        # Conv 1: 16x16x64
        discriminator = Conv2D(64, kernel_size=5, strides=2, padding='same')(img_input)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        #discriminator = LeakyReLU(alpha=0.1)(discriminator)
        discriminator = PReLU()(discriminator)

        # Conv 2:
        discriminator = Conv2D(128, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        #discriminator = LeakyReLU(alpha=0.1)(discriminator)
        discriminator = PReLU()(discriminator)

        # Conv 3:
        discriminator = Conv2D(256, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        #discriminator = LeakyReLU(alpha=0.1)(discriminator)
        discriminator = PReLU()(discriminator)

        # Conv 4:
        discriminator = Conv2D(512, kernel_size=5, strides=2, padding='same')(discriminator)
        discriminator = BatchNormalization(momentum=0.9)(discriminator)
        #discriminator = LeakyReLU(alpha=0.1)(discriminator)
        discriminator = PReLU()(discriminator)

        # FC
        discriminator = Flatten()(discriminator)

        # Concatenate
        merged_layer = Concatenate()([discriminator, bvec, bathy_mean])
        embedding = Dense(512, activation='linear')(merged_layer)
        embedding = PReLU()(embedding)

        # Output Discriminator
        discriminator = Dense(1, activation='sigmoid')(embedding)

        discr = Model(inputs=[img_input, bpatch, bathy_mean], outputs=discriminator, name='discriminator')

        # Recognition
        q_net = Dense(512, activation='linear')(embedding)
        q_net = PReLU()(q_net)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        qaux = Model(inputs=[img_input, bpatch, bathy_mean], outputs=label, name='q_net_aux')

        return discr, qaux

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset

        cached_images = '/data/bathy_training/cache_images_ohara_07.npz'
        # cached bathymetry
        cached_bpatches = '/data/bathy_training/cache_raw_bpatches_ohara_07.npz'
        all_bpatches = '/data/bathy_training/all_raw_bpatches_ohara_07.npz'
        # load dataset images and corresponding bathy patches
        data =  np.load(cached_images)
        Ximg_train = data['xtrain']
        data = np.load(cached_bpatches)
        Xbathy_train = data['xtrain']
        Xbathy_train_means = np.mean(Xbathy_train,axis=(1,2))
        print("shape Xbathy_train_means ", Xbathy_train_means.shape)
        for k in np.arange(Xbathy_train.shape[0]):
            Xbathy_train[k,:,:,0] = Xbathy_train[k,:,:,0] - Xbathy_train_means[k]
        Xcoords_train = data['xtrain_coords']
        # load all bathy patches
        data = np.load(all_bpatches)
        Xbathy_all = data['xtrain']
        Xbathy_all_means = np.mean(Xbathy_all,axis=(1,2))
        for k in np.arange(Xbathy_all.shape[0]):
            Xbathy_all[k,:,:,0] = Xbathy_all[k,:,:,0] - Xbathy_all_means[k]
        Xcoords_all = data['xtrain_coords']
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

            # Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([gen_input[:actual_batch_size], bpatches, bp_means])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, bpatches, bp_means], valid[:actual_batch_size]-np.abs(np.random.normal(0,0.05,(actual_batch_size,1))))
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, bpatches, bp_means], fake[:actual_batch_size]+np.abs(np.random.normal(0,0.05,(actual_batch_size,1))))
            #d_loss_real = self.discriminator.train_on_batch([imgs, bpatches, bp_means], valid[:actual_batch_size])
            #d_loss_fake = self.discriminator.train_on_batch([gen_imgs, bpatches, bp_means], fake[:actual_batch_size])

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and Q-network
            # ---------------------

            # Condition on labels
            # on the same batch used by the discriminator
            #g_loss = self.combined.train_on_batch([gen_input[:actual_batch_size], bpatches, bp_means], [valid[:actual_batch_size], sampled_labels[:actual_batch_size]])

            # this is selecting random patches from a box around dive, presumably representative bathy
            idx = np.random.randint(0, Xbathy_all.shape[0], batch_size)
            random_bathy = Xbathy_all[idx]
            random_bathy_means = Xbathy_all_means[idx]
            # Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
            # Train the generator
            g_loss = self.combined.train_on_batch([gen_input, random_bathy, random_bathy_means], [valid, sampled_labels])



            # Plot the progress
            print ("%d [D loss: %.4f, acc.: %.2f%%] [Q loss: %.4f] [G loss: %3f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            d_loss_hist.append(d_loss[0])
            g_loss_hist.append(g_loss[0])
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, Xbathy_train, Xbathy_train_means)
                self.sample_images_fixed_bathy(epoch, Xbathy_train, Xbathy_train_means)
                self.sample_q_spatially(epoch,Ximg_train, Xbathy_train, Xbathy_train_means)
                # plotting the metrics
                plt.plot(d_loss_hist)
                plt.plot(g_loss_hist)
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Discriminator', 'Adversarial'], loc='best')
                plt.show()
                plt.savefig("metrics_icgan/metrics.png")
                plt.close()

    def sample_images(self, epoch, Xbathy_samples, Xbathy_samples_means):
        # save sample images to disk
        r, c = 6, self.num_classes
        fig, axs = plt.subplots(r, 2*c)


        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(r)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)

            gen_input = np.concatenate((sampled_noise, label), axis=1)

            idx = np.random.randint(0, Xbathy_samples.shape[0], r)
            random_bathy = Xbathy_samples[idx]
            random_bathy_means = Xbathy_samples_means[idx]

            gen_imgs = self.generator.predict([gen_input,random_bathy, random_bathy_means])
            label_pred =  self.auxiliary.predict([gen_imgs,random_bathy, random_bathy_means])

            for j in range(r):
                axs[j,2*i].imshow(gen_imgs[j,::])
                #axs[j,2*i].set_title(("{:0.1f}".format(k) for k in label_pred[j,:]), fontdict={'fontsize':8})
                axs[j,2*i+1].imshow(random_bathy[j,:,:,0])
                axs[j,2*i+1].set_title("d%.1f" % (random_bathy_means[j]), fontdict={'fontsize':8})
                axs[j,2*i].axis('off')
                axs[j,2*i+1].axis('off')


        fig.savefig("images_icgan/%d.png" % epoch)
        plt.close()


    def sample_images_fixed_bathy(self, epoch, Xbathy_samples, Xbathy_samples_means):
        # save sample images to disk
        r, c = 3, self.num_classes
        fig, axs = plt.subplots(2*r, 2*c)

        sampled_noise, _ = self.sample_generator_input(r)
        idx = np.random.randint(0, Xbathy_samples.shape[0], r)
        random_bathy = Xbathy_samples[idx]
        random_bathy_means = Xbathy_samples_means[idx]
        for i in range(c):

            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)

            gen_input = np.concatenate((sampled_noise, label), axis=1)

            gen_imgs = self.generator.predict([gen_input,random_bathy, random_bathy_means])
            label_pred =  self.auxiliary.predict([gen_imgs,random_bathy, random_bathy_means])
            #print("{:0.1f}".format(k) for k in label_pred)
            #print(label_pred)

            for j in range(r):
                axs[2*j,2*i].imshow(gen_imgs[j,::])
                #axs[j,2*i].set_title(("{:0.1f}".format(k) for k in label_pred[j,:]), fontdict={'fontsize':8})
                #axs[j,2*i].set_title("p {:.1f}".format(label_pred[j,:]), fontdict={'fontsize':8})
                #axs[j,2*i].set_title("p%.1f" % (label_pred[j,:]), fontdict={'fontsize':8})
                axs[2*j,2*i+1].imshow(random_bathy[j,:,:,0])
                axs[2*j,2*i+1].set_title("d%.1f" % (random_bathy_means[j]), fontdict={'fontsize':8})
                axs[2*j,2*i].axis('off')
                axs[2*j,2*i+1].axis('off')
                axs[2*j+1,2*i].bar(np.arange(self.num_classes),label_pred[j,:],align='center')
                axs[2*j+1,2*i].set_ylim([0,1])
                axs[2*j+1,2*i].axis('off')
                axs[2*j+1,2*i+1].axis('off')


        fig.savefig("images_icgan_fixed_bathy/fixed_bathy_%d.png" % epoch)
        plt.close()

    def sample_q_spatially(self,epoch, Ximg_samples, Xbathy_samples, Xbathy_samples_means, Xcoords):
        label_pred = self.auxiliary.predict([Ximg_samples, Xbathy_samples, Xbathy_samples_means])
        # max index per prediction
        ml_label=np.argmax(label_pred,axis=1)
        # plot with Xcoords and max index
        plt.scatter(Xcoords[:,0],Xcoords[:,1],c=ml_label)
        ig.savefig("spatial_icgan/spatial_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    infocgan = INFOCGAN()
    infocgan.train(epochs=300000, batch_size=32, sample_interval=200)
