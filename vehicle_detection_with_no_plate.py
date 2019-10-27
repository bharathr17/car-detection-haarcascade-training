# -*- coding: utf-8 -*-

import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from PIL import ImageFont, ImageDraw 
import numpy as np
 
def imgToText(im):
#    im = im.filter(ImageFilter.MedianFilter())
#    enhancer = ImageEnhance.Contrast(im)
#    im = enhancer.enhance(10)
#    #im = im.convert('1')
#    im.save("english_pp.jpg")
    
#    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd ="D:/C-Drive/Tesseract-OCR/tesseract.exe"
    text = pytesseract.image_to_string(im,lang="ben")
    #text = pytesseract.image_to_string(Image.open("english_pp.jpg"),lang="eng")
    print(text)
    return text


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


 


#cascade_src = 'E:/PROJECT ALL/kaggle/project/car detection haarcascade training raju/classifier/cascade.xml'
#cascade_src_no_plate= 'E:/PROJECT ALL/kaggle/project/car number plate detection haarcascade training raju/classifier/cascade.xml'

#video_src = "E:/PROJECT ALL/kaggle/project/car video/VID7.mp4" #laptop
#outputpath='E:/PROJECT ALL/kaggle/project/dataExtract/VID7/'
 



cascade_src = 'D:/PROJECTS/Python/car-detection-haarcascade-training/classifier/cascade.xml'
cascade_src_no_plate= 'D:/PROJECTS/Python/car-number-plate-detection-haarcascade-training/classifier/cascade.xml'


 
video_src = 'D:/PROJECTS/Python/car video/VID7.mp4'
outputpath = 'D:/PROJECTS/Python/dataExtract/VID7/'
# 
 

 
 

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
        if(crop_img.shape[1]>100):
            crop_img = image_resize(crop_img, height = 100)
            s=imgToText(crop_img)
            Img = Image.fromarray(img)  
            ImgD = ImageDraw.Draw(Img)  
            font = ImageFont.truetype("Nikosh.ttf", 20) 
            ImgD.text( (x1 ,y1), s, font=font,fill=(255,255,0,255) ) 
#           
        file_output_path = os.path.join(outputpath+  "-"+str(count)+'.jpg')
        directory = os.path.dirname(file_output_path)
        
 
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
 
        
        
#        cv2.imwrite(file_output_path, crop_img)
        count+=1
 
#    cv2.imshow('video', img)
    if( (Img)!='None'):
        img = cv2.cvtColor(np.asarray(Img),0)

    cv2.imshow('img', img)
#    cv2.imshow('img',cv2.cvtColor(img,cv2.COLOR_BAYER_GR2RGB ) )
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()