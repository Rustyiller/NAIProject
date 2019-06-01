import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_frontalface_alt2.xml')
#ustawienie śledzenia obrazu z kamery
cap = cv2.VideoCapture(0)

while True:

   #odczytywanie klatek z danej kamery
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x,y,w,h) in faces:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('frame', frame)

    #ustawianie koloru szarego


   # cv2.imshow('gray', gray)

    #wyłączenie programu po kliknięciu q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

