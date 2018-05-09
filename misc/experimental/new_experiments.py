import cv2

def masking_attempt(image):
    fgbg = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1)
    masked_image = fgbg.apply(image)
    masked_image[masked_image==127]=0
    cv2.imshow('masked', masked_image)
    cv2.waitKey(0)
    return masked_image