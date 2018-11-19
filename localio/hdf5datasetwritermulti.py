# import the necessary packages
import h5py
import os


from collections import Mapping, Container
from sys import getsizeof

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
    This is a recursive function that rills down a Python object graph
    like a dictionary holding nested ditionaries with lists of lists
    and tuples and sets.
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, bytes):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r


class HDF5DatasetWriter:
    def __init__(self, Idims, Bdims, navdims, outputPath,
                 bufSize=50):
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
        #print("size of image buffer {}".format(deep_getsizeof(self.buffer["images"], set())))
        #print("size of bpatches buffer {}".format(deep_getsizeof(self.buffer["bpatches"], set())))
        #print("size of pixcoords buffer {}".format(deep_getsizeof(self.buffer["pixcoords"], set())))
        #print("size of utmcoods buffer {}".format(deep_getsizeof(self.buffer["utmcoords"], set())))
        #print("size of buffer {}".format(deep_getsizeof(self.buffer, set())))

        #print("length of bpatches buffer {}".format(len(self.buffer["bpatches"])))

        i = self.idx + len(self.buffer["images"])
        #print("size of hdf5 images store before adding buffer {}".format(deep_getsizeof(self.images, set())))
        self.images[self.idx:i] = self.buffer["images"]
        #print("size of hdf5 images store after adding buffer {}".format(deep_getsizeof(self.images, set())))
        #print("size of hdf5 bpatches store before adding buffer {}".format(deep_getsizeof(self.bpatches, set())))
        self.bpatches[self.idx:i] = self.buffer["bpatches"]
        #print("size of hdf5 bpatches store after adding buffer {}".format(deep_getsizeof(self.bpatches, set())))
        self.pixcoords[self.idx:i] = self.buffer["pixcoords"]
        self.utmcoords[self.idx:i] = self.buffer["utmcoords"]
        self.idx = i
        self.buffer = {"images": [],"bpatches": [], "pixcoords": [], "utmcoords": []}


    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["images"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
