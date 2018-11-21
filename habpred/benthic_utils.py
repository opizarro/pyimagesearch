import os
import cv2
import numpy as np

# benthoz images location
image_dir = "/data/Benthoz2015/"
# images stored as png



def benthic_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def load_cache(fn):
    try:
        npzfile = np.load(fn)
        return npzfile
    except:
        print("cache not available, loading individual images")
        return 0

def benthoz_data():
    # images in set
    n = 9600
    # fraction to be 'test'
    testfrac = 0.0
    rescale_im = 0.125
    #
    ntest = int(n * testfrac)
    ntrain = n - ntest

    imsize = 32

    #xtrain = np.empty((ntrain, 128, 170, 3), dtype='uint8')
    #xtest = np.empty((ntest, 128, 170, 3), dtype='uint8')

    # check if images have been cached into array
    npzfile = load_cache('/data/cacheBenthoz32.npz')
    if not npzfile:

        xtrain = np.zeros((ntrain, imsize, imsize, 3))
        xtest = np.zeros((ntest, imsize, imsize, 3))


        i = 0
        #loop through images
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                #read
                print("processing image number {} name {} ".format(i,filename))
                image = cv2.imread(os.path.join(image_dir,filename))
                #small = cv2.resize(image, (0,0), fx=rescale_im,fy=rescale_im)
                small = cv2.resize(image, (imsize,imsize))
                #print("mean small image {}".format(np.mean(small)))
                #cv2.imwrite("output/aae-benthic/" + filename, small)
                #print("counter i {}".format(i))
                if i < ntrain:
                    # save a train array
                    #print("i {} less than num train samps {}".format(i,ntrain))
                    xtrain[i,:,:,:] = benthic_process(small)
                elif i >= ntrain and i < n:
                    #print("i {} equal or greater than {} and less than {}".format(i,ntrain,n))
                    xtest[i-ntrain,:,:,:] = benthic_process(small)
                else:
                    #print("mean xtrain {}".format(np.mean(xtrain)))
                    break
                i += 1

        np.savez('/data/cacheBenthoz32',xtrain=xtrain, xtest=xtest)
    else:
        xtrain = npzfile['xtrain']
        xtest = npzfile['xtest']
    return xtrain, xtest

def bathy_data():

    cached_bpatches = '/data/bathy_training/cache_raw_bpatches_ohara_07.npz'

    # load dataset

    data = np.load(cached_bpatches)
    xtrain = data['xtrain']
    xtest = []

    return xtrain, xtest

def benthic_img_data():

    cached_images = '/data/bathy_training/cache_images_ohara_07.npz'
    data =  np.load(cached_images)
    xtrain = data['xtrain']
    xtest = []

    return xtrain, xtest

def load_cached_training_npy(cached_location):

    data =  np.load(cached_location)

    return data
