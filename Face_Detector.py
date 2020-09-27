import cv2 

#imported from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')

#Convert to Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detecting Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#print(face_coordinates)  [[ 55  49 115 115]]
 
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Picture', img)
cv2.waitKey()


