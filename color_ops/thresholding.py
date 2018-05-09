import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())
 
# load the image from disk
img = cv2.imread(args["image"], 0)
img = cv2.medianBlur(img,5)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.bitwise_not(img)

thresh = cv2.threshold(img, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'OTSU Thresh']
images = [img, th1, th2, th3, thresh]

cv2.imwrite('global_threshold.png', th1)
cv2.imwrite('adaptive_mean.png', th2)
cv2.imwrite('adaptive_gaussian.png', th3)
cv2.imwrite('OTSU_threshold.png', thresh)


# blur = cv2.GaussianBlur(th2, (3, 3), 0)
# edged = cv2.Canny(blur, 20, 100)
# cv2.imwrite('edged2.png', edged)

for i in xrange(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

