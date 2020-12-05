# ********************** program for webcam detection *****************************

import cv2 as cv

# load cascade file
cascade_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv.CascadeClassifier('haarcascade_eye.xml')

# capture video camera
cam = cv.VideoCapture(0)

while True:
    # use to read cam images
    ret, frame = cam.read()

    # converting images in gray scale
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect face
    faces = cascade_face.detectMultiScale(grey, 1.1, 4)
    eye = cascade_eye.detectMultiScale(grey, 1.1, 1)

    # draw rectangle around the face
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (50, 155, 1, 0), 3)
    # for eye rectangle
    for (x, y, w, h) in eye:
        cv.rectangle(frame, (x,y), (x+w, y+h), (50, 155, 1, 0), 1)

    # open camera and frame name
    cv.imshow('MyImage', frame)

# to close camera window
    k = cv.waitKey(30)
    if k == 27:
        break

cam.release()



