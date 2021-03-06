import cv2 
from random import randrange

#imported trained data from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Use VideoCapture(0) to use your default webcam
webcam = cv2.VideoCapture(2)

# To iterate all the frames 
while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 4)

    cv2.imshow('Webcam', frame)
    key = cv2.waitKey(1)
    # Ascii value of Q=81 and q=113
    if key == 81 or key == 113:
        break
    #webcam.release()





