import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
from PIL import ImageFont, ImageDraw, Image  

from scipy import ndimage

 
outputpath="E:/PROJECT ALL/kaggle/project/DatasetBanglaDigitFinal/AutoGenaratedDataset/"
 
#outputpath="D:/PROJECTS/Python/car video/found7/crop/number/digits/"


base_size=28,28  
base=np.zeros(base_size,dtype=np.uint8)
base[:]=255

#১২৩৪৫৬৭৮৯০
count=0;
digitsEn=[ "0","1","2","3","4","5","6","7","8","9"]
digits=[ "০","১","২","৩","৪","৫","৬","৭","৮","৯"]


for d in digits:
    for i in range(20,30):#size
        for angle in range(-5,5):
        
             # Pass the image to PIL  
            pil_im = Image.fromarray(base)  
            draw = ImageDraw.Draw(pil_im)  
        #    font = ImageFont.truetype("Bangla_arial.ttf", 24) 
        #    font = ImageFont.truetype("Siyamrupali.ttf", 24) 
            font = ImageFont.truetype("Nikosh.ttf", i) 
            draw.text( (8,0 ), d, font=font)  
            pil_im = pil_im.rotate(angle, expand=False, fillcolor="white")
            
            ii=digits.index(d)
            
            file_output_path = os.path.join(outputpath+digitsEn[ii]+"/"+str(count)+"-"+str(i)+"-"+str(angle)+'.jpg')
            directory = os.path.dirname(file_output_path)
                
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
                
             
            cv2.imwrite( file_output_path, np.array(pil_im)) 
            count+=1
             
         
     
 
 