# import the necessary packages
import numpy as np
import cv2
import imutils

class CropRotFlipPreprocessor:
    # generates one version of the image doing a random square crop
    # along the long dimension and then randomly applying one of 4 rotations and horizontal flips
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
            h1 = int(deltah*np.random.random)
            h2 = deltah-h1

            coords = [0, h1, self.width, h - h2]


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
            w1 = int(deltaw * np.random.random)
            w2 = deltaw - w1
            coords = [w1, 0, w - w2, self.height]

        # if the image is already square
        else:
            coords = [0, 0, self.width, self.height]


        # loop over the coordinates, extract each of the crops
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            image = image[startY:endY, startX:endX]
            print("cropped image shape {}".format(image.shape))
            #image = cv2.resize(crop, (self.width, self.height),
                              #interpolation=self.inter)


        # check if rotations need to be applied
        if self.rots:
            # rotate by one of four angles 0, 90, 180 and 270 degrees
            rot_angles = [0, 90, 180, 270]
            angle = np.random.choice(rot_angles)
            image = imutils.rotate(image,angle)

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flip
            if np.random.random > 0.5:
                image = cv2.flip(image,1)


        # return the set of crops
        # print("number of crops {}".format(len(crops)))
        # # display all crops
        # for (i, img) in enumerate(crops):
        #     cv2.imshow("crops", img)
        #     cv2.waitKey(0)
        return image
