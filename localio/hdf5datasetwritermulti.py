# import the necessary packages
import h5py
import os
import numpy as np

from collections import Mapping, Container
from sys import getsizeof


class HDF5DatasetWriter:
    def __init__(self, Idims, Bdims, navdims, outputPath,
                 bufSize=1000):
                # check to see if the output path exists, and if so, raise
                # an exception
                if os.path.exists(outputPath):
                    raise ValueError("The supplied 'outputPath' already "
                        "exists and cannot be overwritten. Manually delete "
                        "the file before continuing.", outputPath)

                # open the HDF5 database for writing and create four datasets:
                # one to store the images/features and another to store the
                #  class labels
                self.db = h5py.File(outputPath, "w")
                self.images = self.db.create_dataset("images", Idims,
                    dtype="i1")
                self.bpatches = self.db.create_dataset("bpatches", Bdims,
                                                   dtype="float")
                self.pixcoords = self.db.create_dataset("pixcoords", navdims,
                                                   dtype="float")
                self.utmcoords = self.db.create_dataset("utmcoords", navdims,
                                                   dtype="float")

                # store the buffer size, then initialize the buffer itself
                # along with the index into the datasets
                self.bufSize = bufSize
                self.buffer = {"images": [], "bpatches": [], "pixcoords": [], "utmcoords": []}
                self.idx = 0

    def add(self, Irows, Brows, Prows, Urows):
        # add the Image, Bathy, Pixelcoords, UTMcoords rows to the buffer
        self.buffer["images"].extend(Irows)
        self.buffer["bpatches"].extend(Brows)
        self.buffer["pixcoords"].extend(Prows)
        self.buffer["utmcoords"].extend(Urows)

        #print("data in buffer")
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["images"]) >= self.bufSize:
           # print("need to flush")
            self.flush()
           # print("flushed")

    def flush(self):
        # write the buffers to disk then reset the buffer
        #print("current index {} and increment {}".format(self.idx,len(self.buffer["images"])))
        i = self.idx + len(self.buffer["images"])
        self.images[self.idx:i] = self.buffer["images"]
        self.bpatches[self.idx:i] = self.buffer["bpatches"]
        self.pixcoords[self.idx:i] = self.buffer["pixcoords"]
        self.utmcoords[self.idx:i] = self.buffer["utmcoords"]
        self.idx = i
        self.buffer = {"images": [],"bpatches": [], "pixcoords": [], "utmcoords": []}


    def close(self, shuffle=False):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["images"]) > 0:
            self.flush()

        # shuffle before closing
        if shuffle:
            print("shuffling hd5f before closing with {} rows".format(len(self.images)))
            indeces = np.arange(len(self.images))
            remixed = np.random.shuffle(indeces)
            self.images = self.images[remixed]
            self.bpatches = self.bpatches[remixed]
            self.pixcoords = self.pixcoords[remixed]
            self.utmcoords = self.utmcoords[remixed]

        # close the dataset
        self.db.close()
