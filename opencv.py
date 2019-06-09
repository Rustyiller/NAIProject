import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_eye.xml')
#ustawienie śledzenia obrazu z kamery

cap = cv2.VideoCapture(0)
low_boundary_blue = np.array([90, 140, 30])
high_boundary_blue = np.array([110, 320, 400])

array = []
maxLineLenght = 30

while True:

   #odczytywanie klatek z danej kamery
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #zmiana koloru obrazu przechwytywanego przez kamere
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #szukanie obiektów w danym zakresie kolorów HSV
    filter = cv2.inRange(hsv, low_boundary_blue, high_boundary_blue)
    res = cv2.bitwise_and(frame, frame, mask=filter)
    blurred = cv2.GaussianBlur(hsv, (3, 3), 0)
    #zczytywanie konturów obiektu wykrywanego przez filtr
    contours, _ = cv2.findContours(filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = face_cascade.detectMultiScale(gray)


    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        # tworzenie okręgu otaczającego wykrywany przedmiot
        (x, y), radius = cv2.minEnclosingCircle(largest)

        if radius > 30:
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            cv2.circle(frame, (int(x), int(y)), int(radius / 12), (0, 0, 255), -1)

        array.append((int(x), int(y)))
        for i in range(1, len(array)):
            cv2.line(frame, array[i - 1], array[i], (0, 0, 255), 3)
            if len(array) > 10:
                if (array[i][1] - array[0][1]) > 200:
                    cv2.putText(frame, 'Ruch w dol', (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                    cv2.LINE_AA)
                if (array[i][1] - array[0][1]) > 100:
                    if (array[i][0] - array[0][0]) > 100:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 200, 200), 2)
                            cv2.putText(frame,'Twarz', (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0, 0, 255), 3, cv2.LINE_AA)
                if (array[i][1] - array[0][1]) > 100:
                    if (array[i][0] - array[0][0]) < -100:
                        cv2.putText(frame, 'Oczy', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                                    cv2.LINE_AA)
                        for (x, y, w, h) in faces:
                            roi_gray = gray[y:y + h, x:x + w]
                            roi_color = frame[y:y + h, x:x + w]
                            eyes = eye_cascade.detectMultiScale(roi_gray)
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                if (array[i][1] - array[0][1]) < -200:
                    cv2.putText(frame, 'Ruch w gore', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                        cv2.LINE_AA)
                if (array[i][0] - array[0][0]) < -200:
                    cv2.putText(frame, 'Ruch w lewo', (0, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                        cv2.LINE_AA)
                if (array[i][0] - array[0][0]) > 200:
                    cv2.putText(frame, 'Ruch w prawo', (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                        cv2.LINE_AA)
        if len(array) > maxLineLenght:
            array.pop(0)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow("filter", filter)
    cv2.imshow('frame', frame)

    #wyłączenie programu po kliknięciu q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
