# -*- coding: utf-8 -*-

import cv2
import os
print(cv2.__version__)

#cascade_src = 'cars.xml'
#cascade_src = 'E:/PROJECT ALL/kaggle/project/OpenCV-Dashcam-Car-Detection/cascade_dir/cascade.xml'
#cascade_src = 'numberPlate.xml'
#cascade_src = 'E:/PROJECT ALL/kaggle/project/car detection haarcascade training raju/classifier/cascade.xml'
#video_src = 'dataset/video1.avi'
#video_src = 'dataset/video2.avi'
#video_src = 'dataset/video3.avi'
#video_src = 'dataset/VID_20191002_095248.avi'
video_src = 'E:/PROJECT ALL/kaggle/project/car video/VID_20191002_095248.mp4'
video_src = 'E:/PROJECT ALL/kaggle/project/car video/VID_20191009_104207.mp4'
video_src = "E:/PROJECT ALL/kaggle/project/car video/VID (8).mp4"
cap = cv2.VideoCapture(video_src)
#car_cascade = cv2.CascadeClassifier(cascade_src)
outputpath = 'E:/PROJECT ALL/kaggle/project/car video/car image 8/'
count=0
while True:
    ret, img = cap.read()
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    cv2.imwrite(os.path.join(outputpath , 'Car-8-'+str(count)+'.jpg'), img)
#    cv2.imwrite("frame%d.jpg" % count, image)
    count+=1
    
   

cv2.destroyAllWindows()