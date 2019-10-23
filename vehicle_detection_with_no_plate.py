# -*- coding: utf-8 -*-

import cv2
import os
print(cv2.__version__)
 

#cascade_src = 'cars.xml'
cascade_src = 'E:/PROJECT ALL/kaggle/project/car detection haarcascade training raju/classifier/cascade.xml'
cascade_src_no_plate= 'E:/PROJECT ALL/kaggle/project/car number plate detection haarcascade training raju/classifier/cascade.xml'
#cascade_src_no_plate='E:/PROJECT ALL/kaggle/project/haarcascade_licence_plate_rus_16stages.xml'
#cascade_src_no_plate='E:/PROJECT ALL/kaggle/project/haarcascade_russian_plate_number.xml'
#cascade_src = 'numberPlate.xml'
#video_src = 'dataset/video1.avi'
#video_src = 'dataset/video2.avi'
#video_src = 'dataset/video3.avi'
video_src = 'E:/PROJECT ALL/kaggle/project/car detection haarcascade training raju/dataset/VID_20191002_095248.avi'
 
video_src = 'D:/PROJECTS/Python/car video/VID_20191002_095248.mp4'
outputpath = 'D:/PROJECTS/Python/car video/plate1/'
 
 
video_src = 'E:/PROJECT ALL/kaggle/project/car video/VID (7).mp4' #laptop
outputpath='E:/PROJECT ALL/kaggle/project/found7/'
 
#video_src = 'D:/PROJECTS/Python/car video/VID_20191002_095248.mp4'
#outputpath = 'D:/PROJECTS/Python/car video/plate/'
 
 

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
no_plate_cascade = cv2.CascadeClassifier(cascade_src_no_plate)
count=0
while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
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
    roi_gray = gray[y1:y1+h1, x1:x1+w1]
    roi_color = img[y1:y1+h1, x1:x1+w1]
    
    plates = no_plate_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in plates:
 
        cv2.rectangle(img,(x1+ex,y1+ey),(x1+ex+ew,y1+ey+eh),(0,255,0),2)
        crop_img = roi_gray[ey:ey+eh,ex:ex+ew]
#           
        file_output_path = os.path.join(outputpath+  "-"+str(count)+'.jpg')
        directory = os.path.dirname(file_output_path)
        
 
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
 
        
        
        cv2.imwrite(file_output_path, crop_img)
        count+=1
 
#    cv2.imshow('video', img)
    cv2.imshow('img', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()