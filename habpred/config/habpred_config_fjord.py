
# define the paths to the images directory
IMAGES_PATH = "/Volumes/Samsung_T3/PROCESSED_DATA/Tasmania200810/r20081007_224547_ohara_07_transect/i20081007_224547_cv"

# define the paths to the bathymetric patches
BATHY_PATH = "/Volumes/Samsung_T3/Tassie_bathy/Tasmania200810/TAFI_provided_data/BathymetryAsTiffs/fort1.tif"
BATHY_UTM_PROJ_PARAMS = '+proj=utm +zone=55G, +south +ellps=WGS84'

# NAV PATH
NAV_PATH ="/Volumes/Samsung_T3/PROCESSED_DATA/Tasmania200810/r20081007_224547_ohara_07_transect/renav20160205/stereo_pose_est.data"

NUM_VAL_IMAGES = 500
NUM_TEST_IMAGES = 500

BPATCH_SIZE = 21
IMAGE_SIZE_ROWS = 1024
IMAGE_SIZE_COLS = 1360
#IMAGE_SIZE_ROWS = 128
#IMAGE_SIZE_COLS = 170
PIXCOORD_SIZE = 2
UTMCOORD_SIZE = 2

# define the path to the output training, validation and testing
# HDF5 files
TRAIN_HDF5 = "/Volumes/Samsung_T3/datasets/bathyvis_ohara07/hdf5/train.hdf5"
#TRAIN_HDF5 = "/Users/opizarro/tmp/train.hdf5"

VAL_HDF5 = "/Volumes/Samsung_T3/datasets/bathyvis_ohara07/hdf5/val.hdf5"
TEST_HDF5 = "/Volumes/Samsung_T3/datasets/bathyvis_ohara07/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/cgan_ae.model"

# define the path to the dataset mean
DATASET_MEAN = "output/habpred_image_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc
OUTPUT_PATH = "output"