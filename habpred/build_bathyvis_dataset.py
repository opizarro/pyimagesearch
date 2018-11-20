# import necessary packages
from affine import Affine
from pyimagesearch.habpred.config import habpred_config as config
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.tricroppreprocessor import TriCropPreprocessor
from pyimagesearch.localio.hdf5datasetwritermulti import HDF5DatasetWriter
from pyimagesearch.utils import renavutils3 as rutil
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
from osgeo import gdal
import pyproj
import sys

# grab the paths to the images
# assumes all images in one directory
#trainImgs = list(paths.list_images(config.IMAGES_PATH))



def extract_bathy_patch(gdal_raster,off_ulx,off_uly,patch_size):
    columns = patch_size
    rows = patch_size
    patch_data = gdal_raster.GetRasterBand(1).ReadAsArray(off_ulx, off_uly, columns, rows)
    return patch_data

def retrieve_pixel_coords(geo_coord,geot_params):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform =  Affine.from_gdal(*geot_params)
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px = np.around(px).astype(int)
    py = np.around(py).astype(int)
    pixel_coord = px, py
    return pixel_coord

def extract_geotransform(bathy_path):
    # test section of input data
    in_ds = gdal.Open(bathy_path)

    print("Driver: {}/{}".format(in_ds.GetDriver().ShortName,
                                in_ds.GetDriver().LongName))
    print("Size is {} x {} x {}".format(in_ds.RasterXSize,
                                       in_ds.RasterYSize,
                                      in_ds.RasterCount))
    print("Projection is {}".format(in_ds.GetProjection()))
    geotransform = in_ds.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    #plt.figure(1)
    #plt.imshow(in_ds.GetRasterBand(1).ReadAsArray(), cmap="nipy_spectral")
    return (in_ds,geotransform)

cp = TriCropPreprocessor(128,128)

# extract image paths and coordinates from stereo_pose_est.data
renav, o_lat, o_lon, ftype = rutil.read_renav(config.NAV_PATH)

# get bathymetry (gdal raster) and geotransform
bathy_ds, geotransform = extract_geotransform(config.BATHY_PATH)
# tranformation from lat lon to utm for bathymetry
utm_proj = pyproj.Proj(config.BATHY_UTM_PROJ_PARAMS)

# split into training, validation and testing
# load image, extract bathy patch
# save to HDF5


# perform sampling from the training set to build the
# testing split from the training data
# this could be done over the pose file instead
split = train_test_split(renav['leftim'], test_size=config.NUM_TEST_IMAGES )
(trainImgs, testImgs) = split

# sample to get validation data
split = train_test_split(trainImgs, test_size=config.NUM_VAL_IMAGES)
(trainImgs, valImgs) = split

# construct list pairing training, validation and testing
# image paths along with corresponding bathy and output HDF5
# files
datasets = [
    ("train", trainImgs, config.TRAIN_HDF5),
    ("val", valImgs, config.VAL_HDF5),
    ("test", testImgs, config.TEST_HDF5)]

# initialize the image preprocessor and the lists of RGB channel averages
#aap = AspectAwarePreprocessor(256,256)
(R,G,B) = ([], [], [])

half_patch = np.floor(config.BPATCH_SIZE/2)



# loop over the dataset tuples
for (dType, Imgnames, outputPath) in datasets:
    # assuming three crops, four rotations and horionral flips
    # is 3 x4 x2 = 24 instances
    num_examples = 24 * len(Imgnames)
    # create HDF5 writer
    print("[INFO] building {} with {} images, bathy patches and georef coords...".format(outputPath, len(Imgnames)))
    writer = HDF5DatasetWriter((num_examples, config.IMAGE_SIZE_ROWS, config.IMAGE_SIZE_COLS, 3),
                               (num_examples, config.BPATCH_SIZE, config.BPATCH_SIZE),
                               (num_examples, config.PIXCOORD_SIZE),
                               outputPath)

    # initialise progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(Imgnames), fd=sys.stdout,
                                   widgets=widgets).start()
    # loop over the image paths
    for (i, Imgname) in enumerate(Imgnames):
        # load the image and process it
        Imgpath = os.path.join(config.IMAGES_PATH, Imgname)
        #print(Imgpath)
        image = cv2.imread(Imgpath)
        #image = aap.preprocess(image)


        # given coordinates, extract bathy patch for image
        # get coordinates in lat lon
        x, y = utm_proj(renav['longitude'][i], renav['latitude'][i])
        # convert UTM x,y coords into pixel coords
        px, py = retrieve_pixel_coords([x, y], list(geotransform))
        # calculate offsets
        off_x = int(np.round(px - half_patch))
        off_y = int(np.round(py - half_patch))

        bathy_patch = extract_bathy_patch(bathy_ds, off_x, off_y, config.BPATCH_SIZE)
        #print("shape of bathy patch {}".format(bathy_patch.shape))
        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image to the HDF5 dataset
        # and bathy patch
        # and coordinates in pixels (px, py) and UTM (x,y) for image
        #print(" adding {} to buffer. image shape {}".format(i,image.shape))
        crops = cp.preprocess(image)
        for crop in crops:
            writer.add([crop], [bathy_patch], [(px,py)], [(x,y)])

        pbar.update(i)
        #if i%200 == 0 :
        #    print("added data {}".format(i))

    # close the HDF5 writer
    pbar.finish()
    writer.close(shuffle=True)

# construct a dictionary of averages, the serialise the means to a
# JSON file
print("[INFO] serialising means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

