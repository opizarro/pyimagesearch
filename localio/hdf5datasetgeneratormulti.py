# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py
import keras

class HDF5DatasetGeneratorMulti(keras.utils.Sequence):

    def __init__(self, dbPath, batchSize, preprocessors=None,
                 aug=None):
        # store the batch size, preprocessors, and data augmentor,
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["images"].shape[0]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.numImages / self.batchSize))

    def __getitem__(self, index):
        'Generate one batch of data'
        # generate indexes of the batch
        i = index
        images = self.db["images"][i: i + self.batchSize]
        bpatches = self.db["bpatches"][i: i + self.batchSize]
        pixcoords = self.db["pixcoords"][i: i + self.batchSize]

        # remove bathy mean and add mean depth as an output
        bpatches_means = np.mean(bpatches, axis=(1, 2))
        # print("shape Xbathy_train_means ", Xbathy_train_means.shape)

        # FIXME THIS IS INNEFICIENT - DONE MULTIPLE TIMES
        for k in np.arange(bpatches.shape[0]):
            bpatches[k, :, :] = bpatches[k, :, :] - bpatches_means[k]

        # check to see if our preprocessors are not None
        if self.preprocessors is not None:
            # initialize the list of processed images
            procImages = []

            # loop over the images
            for image in images:
                # loop over the preprocessors and apply each
                # to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

                # update the list of processed images
                procImages.append(image)

            # update the images in the array to be the processed
            # images
            images = np.array(procImages)
        return (images, bpatches, bpatches_means)


    def get_batch_by_indeces(self, indeces):
        'Generate one batch of data'
        # For HDF5 indeces should be sequential for fast access
        #print("in methods indevces {}".format(indeces))
        batch_indeces = list(np.sort(indeces))
        #print("sorted {}".format(batch_indeces))
        images = self.db["images"][batch_indeces]
        bpatches = self.db["bpatches"][batch_indeces]
        pixcoords = self.db["pixcoords"][batch_indeces]

        # remove bathy mean and add mean depth as an output
        bpatches_means = np.mean(bpatches, axis=(1, 2))
        # print("shape Xbathy_train_means ", Xbathy_train_means.shape)

        # FIXME THIS IS INNEFICIENT - DONE MULTIPLE TIMES
        for k in np.arange(bpatches.shape[0]):
            bpatches[k, :, :] = bpatches[k, :, :] - bpatches_means[k]

        # check to see if our preprocessors are not None
        if self.preprocessors is not None:
            # initialize the list of processed images
            procImages = []

            # loop over the images
            for image in images:
                # loop over the preprocessors and apply each
                # to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

                # update the list of processed images
                procImages.append(image)

            # update the images in the array to be the processed
            # images
            images = np.array(procImages)
        return (images, bpatches, bpatches_means)

    def get_random_batch(self):
        i = np.random.randint(0,self.numImages-self.batchSize)
        images = self.db["images"][i: i + self.batchSize]
        bpatches = self.db["bpatches"][i: i + self.batchSize]
        pixcoords = self.db["pixcoords"][i: i + self.batchSize]

        # remove bathy mean and add mean depth as an output
        bpatches_means = np.mean(bpatches, axis=(1, 2))
        # print("shape Xbathy_train_means ", Xbathy_train_means.shape)

        # FIXME THIS IS INNEFICIENT - DONE MULTIPLE TIMES
        for k in np.arange(bpatches.shape[0]):
            bpatches[k, :, :] = bpatches[k, :, :] - bpatches_means[k]

        # check to see if our preprocessors are not None
        if self.preprocessors is not None:
            # initialize the list of processed images
            procImages = []

            # loop over the images
            for image in images:
                # loop over the preprocessors and apply each
                # to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

                # update the list of processed images
                procImages.append(image)

            # update the images in the array to be the processed
            # images
            images = np.array(procImages)
        return (images, bpatches, bpatches_means)


    def close(self):
        # close the database
        self.db.close()
