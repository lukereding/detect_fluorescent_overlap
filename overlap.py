import cv2
import numpy as np

from matplotlib import pyplot as plt

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

from os.path import isfile, join
from os import listdir
import sys
import argparse
import pandas as pd
import subprocess


def detect_images(img):
    """Find which input image is in which channel."""
    max_color_per_row = np.max(img, axis=0)
    blue, green, red = np.max(max_color_per_row, axis=0)
    thresh = 150
    if green > thresh and red < thresh and blue < thresh:
        return "green"
    elif red > thresh and green < thresh and blue < thresh:
        return "red"
    elif blue > thresh and red < thresh and green < thresh:
        return "blue"
    else:
        return "combined"

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", default = ".", help="path folder with images")
    ap.add_argument("-s", "--scale", type=float, default=1.0, help="scaling factor to expand cells once detected")
    args = vars(ap.parse_args())
    return args['path'], args['scale']

def get_images(path):

    # get the jpgs from the path
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(".jpg")]
    print(onlyfiles)
    for img in onlyfiles:
        # read in img
        image = cv2.imread(img)
        if image.shape is None:
            sys.exit("could not read in image")
        color = detect_images(image)
        if color == "green":
            green = image
        elif color == "red":
            red = image
        elif color == "blue":
            blue = image
        elif color == "combined":
            combined = image

    return red, green, blue, combined

def get_thresh(img, min_value = 3):
    # blur
    blurred = cv2.bilateralFilter(img ,3,75,75)
    # convert to gray
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # threshold
    _, thresh = cv2.threshold(gray, min_value, 255, cv2.THRESH_BINARY)
    return thresh


def get_counts(cats):
    categories = pd.Series(cats)
    x = categories.value_counts()
    return x

if __name__ == "__main__":

    # parse arguments (path, scaling_factor)
    path, scaling_factor = get_args()

    # read in photo, find out which which photo is which
    red, green, blue, combined = get_images(path)

    # convert red and green images to gray for later
    red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

    # initialize lists
    red_list = []
    green_list = []
    cat = []
    centers = []

    thresh = get_thresh(img = blue, min_value = 5)

    cv2.imwrite(join(path, "thresh.jpg"), thresh)

    # do the water shedding
    D = ndimage.distance_transform_edt(thresh)
    ### this min_distance parameter is really important
    localMax = peak_local_max(D, indices=False, min_distance= 5, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)


    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(red_gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) > 50:

            # find the average intensity of the red stain contour
            mask = np.zeros(red_gray.shape,np.uint8)
            # get the enclosing circle of the contour
            ((x, y), r) = cv2.minEnclosingCircle(c)
            # draw the circle on an empty image to ask as a mask
            cv2.circle(mask, (int(x), int(y)), int(r*scaling_factor), 1, thickness=-1)

            # get average strength of the red and green signals
            avg_red = cv2.mean(red_gray, mask = mask)[0]
            avg_green = cv2.mean(green_gray, mask = mask)[0]

            # add to lists
            red_list.append(avg_red)
            green_list.append(avg_green)
            centers.append((int(x), int(y)))

            # make into categories
            if avg_green > 15 and avg_red > 14:
                cat.append("both")
            elif avg_green <= 15 and avg_red > 14:
                cat.append("red")
            elif avg_green > 15 and avg_red <= 14:
                cat.append("green")
            else:
                cat.append("nada")

            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(blue, (int(x), int(y)), int(r),(150, 150 ,150), 2)
            cv2.circle(combined, (int(x), int(y)), int(r),(150, 150 , 150), 2)
            if cat[-1] == "both":
                cv2.circle(combined, (int(x), int(y)), int(r),(255, 255 ,255), 2)
                cv2.circle(red, (int(x), int(y)), int(r),(255, 255 ,255), 2)
                cv2.circle(green, (int(x), int(y)), int(r),(255, 255 ,255), 2)
            elif cat[-1] == "red":
                cv2.circle(combined, (int(x), int(y)), int(r),(23, 50 ,255), 2)
                cv2.circle(red, (int(x), int(y)), int(r),(23, 50 ,255), 2)
            elif cat[-1] == "green":
                cv2.circle(combined, (int(x), int(y)), int(r),(14, 255 ,20), 2)
                cv2.circle(green, (int(x), int(y)), int(r),(14, 255 ,20), 2)
            # else:
            #     cv2.circle(combined, (int(x), int(y)), int(r),(150, 150 ,150), 2)
    #         cv2.putText(combined, "{}: {}".format(label, cat[-1]), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (29, 40, 200), 2)

    print("number of nuclei identified: {}".format(len(centers)))

    cv2.imwrite(join(path, "results.jpg"), combined)
    cv2.imwrite(join(path, "red_out.jpg"), red)
    cv2.imwrite(join(path, "green_out.jpg"), green)
    cv2.imwrite(join(path, "blue_out.jpg"), blue)

    results = get_counts(cat)
    print(results)

    # open the result.jpg file
    # bash_command = "open " + join(path, "results.jpg")
    # subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
