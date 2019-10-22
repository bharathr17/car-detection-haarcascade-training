import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

img = cv.imread('C:/Users/Raju/Desktop/15704dsf73130667.jpg',0)
img = cv.imread('D:/PROJECTS/Python/ruff/-4.jpg',0)

#inputpath="D:/PROJECTS/Python/car-number-plate-detection-haarcascade-training/p/"
inputpath="D:/PROJECTS/Python/ruff/"
outputpath="D:/PROJECTS/Python/ruff2/"

count=0


for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
        
        img = cv.imread(inputpath+file,0)
#        img = cv.imread('D:/PROJECTS/Python/ruff/-4.jpg',0)
        
        
        
        edges = cv.Canny(img,100,200)
#        edges1 = cv.Canny(edges,100,200)
        x,y=edges.shape
         
        totalrow = [0]*(x)
        totalclm = [0]*(y)
        for row in range(x):
            for clm in range(y):
                print(edges[row,clm])
                totalrow[row]+=edges[row,clm]
                totalclm[clm]+=edges[row,clm]
         
        
        mid=int(x/2)
         
       
        x1Min=500
       
        x1Index=0
        
        for row in range(x):
 
            if(row < x  ):
                print(row)
                if( totalrow[row] < x1Min):
                    x1Min= totalrow[row]
                    x1Index=row
                
 
         
        
        
            
        
#        mid=int(y/2)
#        y1=mid-1
#        y2=mid+1
#        y1Max=-1
#        y2Max=-1
#        y1Index=0
#        y2Index=0
#        for row in range(mid):
#            if(y1 >= 0):
#                if( totalclm[y1] > y1Max):
#                    y1Max= totalclm[y1]
#                    y1Index=y1
#            if(y2 < y  ):
#                print(y2)
#                if( totalclm[y2] > y2Max):
#                    y2Max= totalclm[y2]
#                    y2Index=y2
#                
#            y1=y1-1
#            y2=y2+1
#            
#         
            
            
        print( str( x1Index)+", " )
       
         
        
        crop_img=img[x1Index:x,0:y]
        
        
        
        file_output_path = os.path.join(outputpath+'-'+str(count)+'.jpg')
        directory = os.path.dirname(file_output_path)
        
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        
        
        cv.imwrite(file_output_path, crop_img)
        count+=1




#x1Index
#y1Index
#x2Index
#y2Index
    
    
    

#print(edges[3,4])
# 
#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()