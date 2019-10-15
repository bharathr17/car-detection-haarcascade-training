# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

#cascade_src = 'cars.xml'
#cascade_src = 'E:/PROJECT ALL/kaggle/project/OpenCV-Dashcam-Car-Detection/cascade_dir/cascade.xml'
#cascade_src = 'numberPlate.xml'
cascade_src = 'E:/PROJECT ALL/kaggle/project/car detection haarcascade training raju/classifier/cascade.xml'
#video_src = 'dataset/video1.avi'
#video_src = 'dataset/video2.avi'
#video_src = 'dataset/video3.avi'
#video_src = 'dataset/VID_20191002_095248.avi'
video_src = 'E:/PROJECT ALL/kaggle/project/car video/VID_20191002_095248.mp4'
video_src = 'E:/PROJECT ALL/kaggle/project/car video/VID_20191009_104207.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1,3)
    x1=0
    y1=0
    w1=0
    h1=0
    for (x,y,w,h) in cars:
        if((w1*h1)<(w*h)):
            x1=x
            y1=y
            w1=w
            h1=h
#        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()