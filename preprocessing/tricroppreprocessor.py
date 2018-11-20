# import the necessary packages
import numpy as np
import cv2
import imutils

class TriCropPreprocessor:
    def __init__(self, width, height, horiz=True, rots=True, inter=cv2.INTER_AREA):
        # store the target image width, height, whether or not
        # horizontal flips should be included, along with the
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.rots = rots
        self.inter = inter

    def preprocess(self, image):
        # initiliaze the list of crops
        crops = []

        # grab the dimensions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smaller than the height, then resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # dimension
        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.inter)
            (h, w) = image.shape[:2]
            # grab the width and height of the scaled image then use these
            # dimensions to define the corners of the image based
            deltah = h-self.height
            coords = [
                [0, 0, self.width, self.height],
                [0, deltah//2, self.width, h - deltah//2],
                [0, deltah, self.width, h]]

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
        elif w > h:
            image = imutils.resize(image, height=self.height,
                                   inter=self.inter)
            # grab the width and height of the scaled image then use these
            # dimensions to define the corners of the image based
            (h, w) = image.shape[:2]
            deltaw = w-self.width

            coords = [
                [0, 0, self.width, self.height],
                [deltaw//2, 0, w - deltaw//2, self.height],
                [deltaw, 0, w, self.height]]
        # if the image is already square
        else:
            coords = [
                [0, 0, self.width, self.height]]


        # loop over the coordinates, extract each of the crops
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        # check if rotations need to be applied
        if self.rots:
            # rotate each crop by 90, 180 and 270 degrees
            rot_angles = [90, 180, 270]
            rotations = [imutils.rotate(c,angle) for c in crops for angle in rot_angles]
            crops.extend(rotations)
        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c,1) for c in crops]
            crops.extend(mirrors)

        # return the set of crops
        # print("number of crops {}".format(len(crops)))
        # # display all crops
        # for (i, img) in enumerate(crops):
        #     cv2.imshow("crops", img)
        #     cv2.waitKey(0)
        return np.array(crops)
