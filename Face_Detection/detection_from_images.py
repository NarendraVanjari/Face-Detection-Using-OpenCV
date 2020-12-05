
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('images/img1.jpeg')

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img, 1.1, 3)
eye = eye_cascade.detectMultiScale(img, 1.1, 3)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (50, 155, 1, 0), 3)

for (x, y, w, h) in eye:
    cv.rectangle(img, (x,y), (x+w, y+h), (50, 155, 1, 0), 3)

cv.imshow('Hello', img)

cv.waitKey()


