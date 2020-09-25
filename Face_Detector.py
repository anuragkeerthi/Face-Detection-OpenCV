import cv2 

#imported from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')

cv2.imshow('Picture', img)
cv2.waitKey()