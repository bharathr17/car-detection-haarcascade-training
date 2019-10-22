import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

#img = cv.imread('C:/Users/Raju/Desktop/15704dsf73130667.jpg',0)
inputpath="E:/PROJECT ALL/kaggle/project/found1/"
outputpath="E:/PROJECT ALL/kaggle/project/found1/crop/"

count=0


for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
        
        img = cv.imread(inputpath+file,0)
 
        
        
        
        
        edges = cv.Canny(img,100,200)
#        %varexp --imshow edges
#        edges1 = cv.Canny(edges,100,200)
        x,y=edges.shape
         
        totalrow = [0]*(x)
        totalclm = [0]*(y)
        for row in range(x):
            for clm in range(y):
                print(edges[row,clm])
                totalrow[row]+=edges[row,clm]
                totalclm[clm]+=edges[row,clm]
                
#        %varexp --plot totalclm
         
        
        mid=int(x/2)
        x1=mid-1
        x2=mid+1
        x1Max=-1
        x2Max=-1
        x1Index=0
        x2Index=0
        for row in range(mid):
            if(x1 >= 0):
                if( totalrow[x1] > x1Max):
                    x1Max= totalrow[x1]
                    x1Index=x1
            if(x2 < x  ):
                print(x2)
                if( totalrow[x2] > x2Max):
                    x2Max= totalrow[x2]
                    x2Index=x2
                
            x1=x1-1
            x2=x2+1
            
         
        
        
            
        
        mid=int(y/2)
        y1=mid-1
        y2=mid+1
        y1Max=-1
        y2Max=-1
        y1Index=0
        y2Index=0
        for row in range(mid):
            if(y1 >= 0):
                if( totalclm[y1] > y1Max):
                    y1Max= totalclm[y1]
                    y1Index=y1
            if(y2 < y  ):
                print(y2)
                if( totalclm[y2] > y2Max):
                    y2Max= totalclm[y2]
                    y2Index=y2
                
            y1=y1-1
            y2=y2+1
            
         
            
            
        print( str( x1Index)+", "+str(y1Index))
        print( str( x2Index)+", "+str(y2Index))
         
        
        crop_img=img[x1Index:x2Index,y1Index:y2Index]
        
        
        
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
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()