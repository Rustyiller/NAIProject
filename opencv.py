import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('casscades/data/haarcascade_eye.xml')
#ustawienie śledzenia obrazu z kamery

cap = cv2.VideoCapture(0)
low_boundary_blue = np.array([90, 140, 30])
high_boundary_blue = np.array([150, 320, 270])

img1 = cv2.imread("eyes.png", 0)
img2 = cv2.imread("ears.png", 0)

def add_transparent_image(bg, ov, x, y):
    x -= int(ov.shape[0] / 2)
    y -= int(ov.shape[1] / 2)
    w = x + ov.shape[0]
    h = y + ov.shape[1]

    alpha_ov = ov[: : 3] / 255.0

    alpha_bg = 1.0 - alpha_ov

    for c in range(0, 3):
        try:
            bg[x:w, y:h, c] = (ov[: : c] * alpha_ov + bg[x:w, y:h, c] * alpha_bg)
        except:
            pass


while True:

   #odczytywanie klatek z danej kamery
    ret, frame = cap.read()
    #zmiana koloru obrazu przechwytywanego przez kamere
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #szukanie obiektów w danym zakresie kolorów HSV
    filter = cv2.inRange(hsv, low_boundary_blue, high_boundary_blue)
    #res = cv2.bitwise_and(frame, frame, mask=filter)
    blurred = cv2.GaussianBlur(hsv, (3, 3), 0)
    #zczytywanie konturów obiektu wykrywanego przez filtr
    contours, _ = cv2.findContours(filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (120,200,200), 2)
       #add_transparent_image(frame,img2,x,y)
       #przypisywanie do zmiennej interesującego nas regiony obrazu
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       for (ex, ey, ew, eh) in eyes:
           cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    if len(contours) > 0:


        largest = max(contours, key=cv2.contourArea)
        #tworzenie okręgu otaczającego wykrywany przedmiot
        (x, y), radius = cv2.minEnclosingCircle(largest)

        if radius > 10:

            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            cv2.circle(frame, (int(x), int(y)), int(radius / 12), (0, 0, 255), -1)

    cv2.imshow('hsv', hsv)
    cv2.imshow("filter", filter)
    cv2.imshow('frame', frame)

    #wyłączenie programu po kliknięciu q
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
