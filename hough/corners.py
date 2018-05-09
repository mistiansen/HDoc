import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


def harris(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    return img

def subpixel(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    image[res[:,1],res[:,0]]=[0,0,255]
    image[res[:,3],res[:,2]] = [0,255,0]
    return image

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    result = subpixel(img)
    # result = harris(img)
    images = [result]

    for i in xrange(len(images)):
        plt.subplot(1,len(images),i+1),plt.imshow(images[i],'gray')
        # plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()