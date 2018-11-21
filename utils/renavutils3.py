""" A Bunch of python utilities for processing renav and clustering results.

Author: Daniel Steinberg
        Australian Centre for Field Robotics, The University of Sydney
Date:   10/05/2013

"""
import random
import csv
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdat
from datetime import datetime
#from imdescrip.utils.image import imread_resize


# Register a new CSV dialect
csv.register_dialect('renav', delimiter=' ', lineterminator='\n',
                     skipinitialspace=True)

csv.register_dialect('renav_tom', delimiter=',', lineterminator='\n',
                     skipinitialspace=True)


def read_renav(renav_file):
    """ Read and parse stereo/vehicle_pose_est.data file.

    Arguments:
        renav_file: the renav stereo/vehicle_pose_est.data file

    Returns:
        renav: a dictionary with the following entries, 'record', 'timestamp',
            'latitude', 'longitude', 'Xpos', 'Ypos', 'Zpos', 'Xang', 'Yang',
            'Zang', 'leftim', 'rightim', 'altitude', 'boundrad', 'crosspoint'.
            Each of these containes a list of the elements corresponds to the
            columns in stero/vehicle_pose_est.data. Note, vehicle_pose_est.data
            will have empty 'leftim', 'rightim', 'boundrad' and 'crosspoint'
            entries.
        origin_latitude: latitude of the dive origin.
        origin_longitude: longitude of the dive origin.
        ftype: 'stereo' for stereo_pose_est.data input, 'vehicle' for
            vehicle_pose_est.data input.

    """

    with open(renav_file,"rt") as f:
        csvread = csv.reader(f, 'renav')

        renav = {
                'record'        : [],
                'timestamp'     : [],
                'latitude'      : [],
                'longitude'     : [],
                'Xpos'          : [],
                'Ypos'          : [],
                'Zpos'          : [],
                'Xang'          : [],
                'Yang'          : [],
                'Zang'          : [],
                'leftim'        : [],
                'rightim'       : [],
                'altitude'      : [],
                'boundrad'      : [],
                'crosspoint'    : []
        }

        origin_latitude = None
        origin_longitude = None
        ftype = None

        for row in csvread:

            # skip empty lines
            if not row:
                continue

            if row[0].startswith(' ') or row[0].startswith('\t'):
                continue

            # skip commented or irrelevant lines
            if row[0].startswith('%'):
                continue

            # Get origin
            if row[0].startswith('ORIGIN_LATITUDE'):
                origin_latitude = float(row[1].strip())
                continue
            elif row[0].startswith('ORIGIN_LONGITUDE'):
                origin_longitude = float(row[1].strip())
                continue

            if ftype is None:
                if len(row) == 11:
                    ftype = 'vehicle'
                elif len(row) == 9:
                    ftype = 'v1_vehicle'
                elif len(row) == 13:
                    ftype = 'v1_stereo'
                elif len(row) == 15:
                    ftype = 'stereo'
                else:
                    raise RuntimeError("Unexpected csv file type! Check nav file")

            if (ftype is 'vehicle') or (ftype is 'stereo'):
                renav['record'].append(int(row[0].strip()))
                renav['timestamp'].append(float(row[1].strip()))
                renav['latitude'].append(float(row[2].strip()))
                renav['longitude'].append(float(row[3].strip()))
                renav['Xpos'].append(float(row[4].strip()))
                renav['Ypos'].append(float(row[5].strip()))
                renav['Zpos'].append(float(row[6].strip()))
                renav['Xang'].append(float(row[7].strip()))
                renav['Yang'].append(float(row[8].strip()))
                renav['Zang'].append(float(row[9].strip()))

                if ftype is 'vehicle':
                    renav['altitude'].append(float(row[10].strip()))
                else:
                    renav['leftim'].append(row[10].strip())
                    renav['rightim'].append(row[11].strip())
                    renav['altitude'].append(float(row[12].strip()))
                    renav['boundrad'].append(float(row[13].strip()))
                    renav['crosspoint'].append(bool(row[14].strip()))
            elif (ftype is 'v1_vehicle') or (ftype is 'v1_stereo'):
                renav['record'].append(int(row[0].strip()))
                renav['timestamp'].append(float(row[1].strip()))
                renav['Xpos'].append(float(row[2].strip()))
                renav['Ypos'].append(float(row[3].strip()))
                renav['Zpos'].append(float(row[4].strip()))
                renav['Xang'].append(float(row[5].strip()))
                renav['Yang'].append(float(row[6].strip()))
                renav['Zang'].append(float(row[7].strip()))

                if ftype is 'v1_vehicle':
                    renav['altitude'].append(float(row[8].strip()))
                else:
                    renav['leftim'].append(row[8].strip())
                    renav['rightim'].append(row[9].strip())
                    renav['altitude'].append(float(row[10].strip()))
                    renav['boundrad'].append(float(row[11].strip()))
                    renav['crosspoint'].append(bool(row[12].strip()))


    return renav, origin_latitude, origin_longitude, ftype


def read_tombridge_georef(renav_file):
    """ Read and parse Tom Bridge's GBR2007 csv file.

    Arguments:
        csv georef file: the camera pose file georeferenced

    Returns:
        renav: a dictionary with the following entries, 'record', 'timestamp',
            'latitude', 'longitude', 'Xpos', 'Ypos', 'Zpos', 'Xang', 'Yang',
            'Zang', 'leftim', 'rightim', 'altitude', 'boundrad', 'crosspoint'.
            Each of these containes a list of the elements corresponds to the
            columns in stero/vehicle_pose_est.data. Note, vehicle_pose_est.data
            will have empty 'leftim', 'rightim', 'boundrad' and 'crosspoint'
            entries.
        origin_latitude: latitude of the dive origin.
        origin_longitude: longitude of the dive origin.
        ftype: 'stereo' for stereo_pose_est.data input, 'vehicle' for
            vehicle_pose_est.data input.
FID,EASTING,NORTHING,IDENTNO,TIMESTAMP,LOCALNORTH,LOCALEAST,
DEPTH,ROLL,PITCH,YAW,LEFTIMAGE,RIGHTIMAGE,ALTITUDE,RADIUS,OVERLAP


    """

    with open(renav_file, 'rb') as f:
        csvread = csv.reader(f, 'renav_tom')

        renav = {
                'record'        : [],
                'easting'       : [],
                'northing'      : [],
                'identno'       : [],
                'timestamp'     : [],
                'Xpos'          : [],
                'Ypos'          : [],
                'Zpos'          : [],
                'Xang'          : [],
                'Yang'          : [],
                'Zang'          : [],
                'leftim'        : [],
                'rightim'       : [],
                'altitude'      : [],
                'boundrad'      : [],
                'crosspoint'    : []
        }


        ftype = None

        for row in csvread:

            # skip empty lines
            if not row:
                continue

            if row[0].startswith(' ') or row[0].startswith('\t'):
                continue

            # skip header line
            if row[0].startswith('FID'):
                print('found header')
                continue

            if ftype is None:
                print('row length ' + str(len(row)))
                print(row)
                if len(row) == 16:
                    ftype = 'tomcsv'
                else:
                    raise RuntimeError("Unexpected csv file type! Check nav file")

            if (ftype is 'tomcsv'):
                renav['record'].append(int(row[0].strip()))
                renav['easting'].append(float(row[1].strip()))
                renav['northing'].append(float(row[2].strip()))
                renav['identno'].append(int(row[3].strip()))
                renav['timestamp'].append(float(row[4].strip()))
                renav['Xpos'].append(float(row[5].strip()))
                renav['Ypos'].append(float(row[6].strip()))
                renav['Zpos'].append(float(row[7].strip()))
                renav['Xang'].append(float(row[8].strip()))
                renav['Yang'].append(float(row[9].strip()))
                renav['Zang'].append(float(row[10].strip()))
                renav['leftim'].append(row[11].strip())
                renav['rightim'].append(row[12].strip())
                renav['altitude'].append(float(row[13].strip()))
                renav['boundrad'].append(float(row[14].strip()))
                renav['crosspoint'].append(bool(row[15].strip()))


    return renav, ftype



def read_labels(labelfile):
    """ Parse a cluster label file.

    Arguments:
        labelfile: either the unmatched or the unmatched cluster label files.

    Returns if unmatched label file:
        labels: a dictionary with the following entries; 'grayimage', 'label'.
            Each of these contains a list of the corresponing CSV column
            entries.
    Returns if matched label file:
        labels: a dictionary file with  the following entries in a addition to
            those above; 'record', 'timestamp', 'colimage'.
        ftype: 'matched' input label file is matched to the renav, 'unmatched'
            is the unmatched label file.

    """

    with open(labelfile, 'rb') as f:
        csvread = csv.reader(f, 'renav')

        ftype = None
        labels = {
                'record'    : [],
                'timestamp' : [],
                'grayimage' : [],
                'colimage'  : [],
                'label'     : [],
        }

        for row in csvread:

            # skip commented or irrelevant lines
            if row[0].startswith('%'):
                continue

            # detect file type
            if ftype is None:
                ftype = 'matched' if len(row) > 2 else 'unmatched'

            # parse
            if ftype is 'matched':          # renav matched label files
                labels['record'].append(int(row[0].strip()))
                labels['timestamp'].append(float(row[1].strip()))
                labels['colimage'].append(row[2].strip())
                labels['grayimage'].append(row[3].strip())
                labels['label'].append(int(row[4].strip()))
            elif ftype is 'unmatched':      # Original label file
                labels['grayimage'].append(row[0].strip())
                labels['label'].append(int(row[1].strip()))

    return labels, ftype


def overlay_depth(renav_file, title=''):
    """ Overlay image depth on the AUV's survey pattern.

    Returns:
        A matplotlib figure object.

    """

    renav, _, _, _ = read_renav(renav_file)

    legend = []
    fig = plt.figure()
    ax = fig.add_subplot(111)

    depth = np.array(renav['Zpos']) + np.array(renav['altitude'])
    scat = ax.scatter(renav['Ypos'], renav['Xpos'], c=depth, s=5,
                      cmap='jet', marker='o', edgecolor='none')

    ax.axis('equal')
    ax.grid(True)
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label('Depth (m)')
    cbar.ax.invert_yaxis()
    ax.set_xlabel('Local Eastings (m)')
    ax.set_ylabel('Local Northings (m)')
    ax.set_title(title)

    return fig


def overlay_clusters(stereo_renav, labelfile, title='', showtrack=False,
                     showclusters=True, cmap='gist_rainbow'):
    """ Overlay cluster labels on the AUV's survey pattern.

    Returns:
        A matplotlib figure object.
    """

    renav, _, _, ftyper = read_renav(stereo_renav)
    labels, ftypel = read_labels(labelfile)

    if ftyper is not 'stereo':
        raise ValueError('stereo_pose_est.data file required!')
    if ftypel is not 'matched':
        raise ValueError('labelfile needs to be matched with the vehicle \
                poses!')

    legend = []
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if showtrack is True:
        ax.plot(renav['Ypos'], renav['Xpos'], 'k', alpha=0.5)
        legend.append('Vehicle Track')

    ax.scatter(renav['Ypos'], renav['Xpos'], c=labels['label'], s=5, cmap=cmap,
               marker='o', edgecolor='none')
    legend.append('Cluster Labels')

    ax.axis('equal')
    ax.grid(True)
    ax.set_xlabel('Local Eastings (m)')
    ax.set_ylabel('Local Northings (m)')
    ax.set_title(title)
    ax.legend(legend)

    return fig


def depth_profile(renav_file):
    """ Return a figure of the AUV's depth profile.

    Arguments:
        renav_file: either the stereo or vehicle_pose_est.data renav files.

    Returns:
        a matplotlib figure of the AUV's depth profile.
    """

    renav, _, _, ftype = read_renav(renav_file)

    if (ftype is not 'vehicle') and (ftype is not 'stereo'):
        raise ValueError('vehicle_pose_est.data or stereo_pose_set.data file'
                         ' required!')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Get date axis
    timelist = [datetime.utcfromtimestamp(t) for t in renav['timestamp']]
    ptime = mdat.date2num(timelist)

    # Plot AUV depth
    ax.plot_date(ptime, renav['Zpos'], 'b')

    # Plot seafloor depth
    alt = np.array(renav['altitude'])
    depth = np.array(renav['Zpos']) + alt
    depth = depth[alt > 0]
    ptime = ptime[alt > 0]
    ax.plot_date(ptime, depth, 'r')

    ax.grid(True)
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Time (UTC)')
    ax.legend(['AUV depth', 'Seafloor'])

    return fig


def show_clusters(imfolder, labelfile, nimgs=10, cmap='gist_rainbow'):
    """ Show a mosaic of sample cluster images using a label file.

    Returns:
        A 'RGB' image matrix which is the mosaic.
    """

    labels, ftype = read_labels(labelfile)
    images = labels['grayimage'] if ftype is 'unmatched' else labels['colimage']
    imagepaths = [os.path.join(imfolder, s) for s in images]
    return sample_images(labels['label'], imagepaths, nimgs, cmap=cmap)


def sample_images(labels, imagepaths, nsamples, maxdim=200,
                  cmap='gist_rainbow'):
    """ Show a mosaic of sample cluster images

    Arguments:
        labels: a list of N integer labels.
        imagepaths: a list of N paths to the correposding images.
        nsamples: number of samples to show per cluster.
        maxdim: max size (pixels) of the thumbnails.
        cmap: a matplotlib colour map to use to border the cluster images.

    Returns:
        A 'RGB' image matrix which is the mosaic.
    """

    K = max(labels)
    labels = np.array(labels)

    cm = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=1, vmax=K)
    imarray = []

    for k in range(1, K+1):

        imidx = (labels == k).nonzero()[0]
        nims = len(imidx)

        if nims == 0:
            continue

        limages = [imagepaths[i] for i in imidx]
        simages = random.sample(limages, min(nims, nsamples))

        col = tuple(np.uint8(255*np.array(cm(norm(k))[0:3])))
        cimarray = []

        #for n, imname in enumerate(simages):
        for n in range(nsamples):
            if n < nims:
                img = Image.fromarray(imread_resize(simages[n], maxdim))
                # Get the image size to make blanks
                imsize = img.size
            else:
                assert n > 0, "This cluster has no images and was not skipped!"
                img = Image.new("RGB", imsize)

            bimg = ImageOps.expand(img, border=6, fill=col)
            cimarray.append(np.array(bimg))

        imarray.append(np.concatenate(cimarray, axis=1))

    return np.concatenate(imarray, axis=0)
