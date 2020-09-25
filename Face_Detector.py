import cv2 

#imported from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')

#Convert to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('Picture', grayscaled_img)
#cv2.waitKey()


