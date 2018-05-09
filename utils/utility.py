import sys
import shapely.geometry
import shapely.affinity
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import cv2
import functions
import os

# <------ PROCESSING/DRAWING METHDOS --------> #

def process_several(images, function, **kwargs):
    results = []
    for image in images:
        result = function(image, **kwargs)
        results.append(result)
    # plot_images(results)
    return results 

def draw_several(images, drawing_images, function, **args): # this really only gets used with HoughLines, where we have different images for processing vs. drawing on 
    if not len(drawing_images) == len(images):
        # print("In display_several, number of images and images to draw on are not the same")
        return 
    else:
        drawn_images = []
        for i, image in enumerate(images):
            drawn_image = function(image, drawing_images[i], **args)
            drawn_images.append(drawn_image)
    return drawn_images

def draw_several2(images, drawing_images, function, **args): # this really only gets used with HoughLines, where we have different images for processing vs. drawing on 
    if not len(drawing_images) == len(images):
        # print( "In display_several, number of images and images to draw on are not the same"
        return 
    else:
        drawn_images = []
        for i, image in enumerate(images):
            _, drawn_image = function(image, drawing_images[i], **args)
            drawn_images.append(drawn_image)
    return drawn_images


# <-----  PLOTTING METHODS  ------> #

def plot_images(images, titles = None):
    if len(images) == 1:
        if titles:
            plot_image(images[0], titles[0])
        else:
            plot_image(images[0])
    elif len(images) < 5:
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1), plt.imshow(images[i], 'gray')
            if titles:
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()   
    else:
        for i in range(len(images)):
            plt.subplot(2, int(round(len(images)/2.0)), i+1),plt.imshow(images[i],'gray') # nrows, ncols, plot_index. Arg names don't work for some reason.
            if titles:
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()   

def plot_image(image, title = None):
    plt.imshow(image, 'gray')
    if title:
        plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()

# <------ END FUNCTIONS.PY --------> #

# <-----  FILE PROCESSING METHODS  ------> #

def image_paths():
    images = []
    for i, arg in enumerate(sys.argv):
        if i == 0: continue # because the first arg is the name of the python file
        else:
            images.append(arg)
    return images

def filenames_at_path(path, extension = ".jpg"):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path,file))
    return files

def image_reading(files):
    originals = []
    images = []
    for filename in files:
        image = cv2.imread(filename)
        images.append(image)
        originals.append(image.copy())
    return originals, images

def image_array(image, array_length):
    array = []
    for i in range(0, array_length):
        array.append(image.copy())
    return array

def save_images(names, images):
    for i in range(len(images)):
        cv2.imwrite(names[i], images[i])


# <-----  SHAPELY STUFF (?)  ------> #

def IOU(rect1, rect2):

    print("rect1 in IOU " + str(rect1))
    print("rect2 in IOU " + str(rect2))

    center_x1, center_y1 = rect1[0][:2]
    width1, height1 = rect1[1][:2]
    angle1 = -1.0 * rect1[2]

    center_x2, center_y2 = rect2[0][:2]
    width2, height2 = rect2[1][:2]
    angle2 = -1.0 * rect2[2] # trying this because it seems the original rect was rotated at the wrong angle
    print(center_x2)
    print(center_y2)
    print(width2)
    print(height2)
    print(angle2)

    r1 = RotatedRect(center_x1, center_y1, width1, height1, angle1)
    r2 = RotatedRect(center_x2, center_y2, width2, height2, angle2)

    print("IOU is: " + str(r1.intersection(r2).area/r1.union(r2).area))

    fig = plt.figure(1, figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 133)

    ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
    ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
    ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

    plt.show()

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def union(self, other):
        return self.get_contour().union(other.get_contour())


def shapely_demo():

    # r1 = RotatedRect(10, 15, 15, 10, 30)
    # r2 = RotatedRect(15, 15, 20, 10, 0)

    r1 = RotatedRect(51.0, 71.5, 84.0, 107.0, -0.0)
    r2 = RotatedRect(51.5, 74.0, 71.0, 88.0, -0.0)

    fig = plt.figure(1, figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 133)

    ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
    ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
    ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

    plt.show()


if __name__ == '__main__':
    shapely_demo()