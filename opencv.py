import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_eye.xml')
#ustawienie śledzenia obrazu z kamery
cap = cv2.VideoCapture(0)

while True:

   #odczytywanie klatek z danej kamery
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]

       eyes = eye_cascade.detectMultiScale(roi_gray)
       for (ex, ey, ew, eh) in eyes:
           cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('frame', frame)



    #wyłączenie programu po kliknięciu q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
