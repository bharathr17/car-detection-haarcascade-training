import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


inputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/crop/"
outputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/crop/number/"




#inputpath="D:/PROJECTS/Python/car video/found7/crop/"
#outputpath="D:/PROJECTS/Python/car video/found7/crop/number/"






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
        totalrowchange = [0]*(x)
        totalclm = [0]*(y)
        totalclmchange = [0]*(y)
         
        
        for row in range(x):
            for clm in range(y):
                print(edges[row,clm])
                totalrow[row]+=edges[row,clm]
                totalclm[clm]+=edges[row,clm]
         
        
        for i in range(y-1):
            totalclmchange[i]=abs(totalclm[i]-totalclm[i+1])
            
            
        for i in range(x-1):
            totalrowchange[i]=abs(totalrow[i]-totalrow[i+1])
                

        elementCount=0
        range1=[]
        i=0
        while( i <y-1):

            i1=i
            while((totalclmchange[i]<1)and( i <y-1)):
                i+=1
            if((i-i1)>5):
                range1.append((i1,i))
            
            i+=1;
            
        range1.sort()
        
        ll=len(range1)
        if(ll==0):
             range1.append((0,0))
             range1.append((y,y))
        elif(ll==1):
            if(range1[0][0]<(y/2)):
                range1.append((0,0))
            else:
                range1.append((y,y))
                
        ll=len(range1)
        range1.sort()  
        
        
        
        
        
        
        
        elementCount2=0
        range2=[]
        i=x-1
      
        while( i >0):
            print(totalrowchange[i])
            i1=i
            while((totalrowchange[i]<20)and( i >0)):
                i-=1
            if(abs(i-i1)>3):
                range2.append((i,i1))
                i=-10
            
            i-=1;
            
        range2.sort()
        
        
        
        if(len(range2)>0):
            img=img[int((range2[0][0]+range2[0][1])/2):x,range1[0][1]:range1[ll-1][0]]
        else:
            img=img[int(x/2):x,range1[0][1]:range1[ll-1][0]]
            
        x,y=img.shape
        
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
       
         
        
#        crop_img=img[x1Index:x,0:y]
        crop_img=img
        
        
        
        file_output_path = os.path.join(outputpath+file+'-'+str(count)+'.jpg')
        directory = os.path.dirname(file_output_path)
        
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        
        hh,ww=crop_img.shape
        if(ww>(hh*4)):
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