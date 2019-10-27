import cv2
import numpy as np
import math
import os
from scipy import ndimage
#import isect_segments_bentley_ottmann.poly_point_isect as bot

inputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/"
outputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/crop/"

#inputpath="D:/PROJECTS/Python/car video/found7/"
#outputpath="D:/PROJECTS/Python/car video/found7/crop/"

def lineEquationA_B_C(x1,y1,x2,y2):
    m1=(x2-x1)
    if(m1==0):
        m1=1
    m=(y2-y1)/m1
    A=m
    B=-1
    C=((y1)-(m*x1))
    return A,B,C
    

def lineEquationGetX(x, A,B,C):
    
    return -((A*x)+C)/B
    
 

def lineEquationGetY(y, A,B,C):
    
    return -((B*y)+C)/A
    
   
def distance(x0,y0,A,B,C):
    k0=abs((A*x0)+(B*y0)+(C))
    k1=math.sqrt( (A*A)+(B*B))
    return k0/k1
def getTwoIntersectionPoint(x_1,y_1,x_2,y_2,XXX,YYY):
    pointss=[]
    A,B,C=lineEquationA_B_C(x_1,y_1,x_2,y_2)

    x_=lineEquationGetX(0,A,B,C)
    if(x_>XXX):
        x_=-1
    if(x_<0):
        x_=-2
        
    x_=int(x_)
    x_,0
    if(x_>=0):
        pointss.append((x_,0));
    


    y_=lineEquationGetY(0,A,B,C)
    if(y_>YYY):
        y_=-1
    if(y_<0):
        y_=-2
        
    y_=int(y_) 
    0,y_
    if((y_>=0)):
        pointss.append((0,y_));


    x_=lineEquationGetX(YYY,A,B,C)
    if(x_>XXX):
        x_=-1
    if(x_<0):
        x_=-2
        
    x_=int(x_)
    x_,YYY
    if((x_>=0)):
        pointss.append((x_,yy));


    y_=lineEquationGetY(XXX,A,B,C)
    if(y_>YYY):
        y_=-1
    if(y_<0):
        y_=-2
        
    y_=int(y_) 
    XXX,y_
    if((y_>=0)):
        pointss.append((XXX,y_))
        
        
        
        
    l=len(pointss)
    temp=pointss[l-1]
    i=0
    while(i<l):
        if((temp[0]==pointss[i][0])and (temp[1]==pointss[i][1])):
            pointss.remove(pointss[i])
            i-=1
            
        temp=pointss[i]
        i+=1
        l=len(pointss)
        
    
    return pointss




def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot





def fourPointCrop(image,points_):
    
    cnt = np.array([
            [[ points_[0][1], points_[0][0]]],
             [[   points_[1][1], points_[1][0]]],
             [[   points_[2][1], points_[2][0]]],
            [[   points_[3][1], points_[3][0]]]
            
           
        ])
    
    print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    print("rect: {}".format(rect))
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    print("bounding box: {}".format(box))
#    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    
    img_crop, img_rot = crop_rect(image, rect)
    
#    print("size of original img: {}".format(image.shape))
#    print("size of rotated img: {}".format(img_rot.shape))
#    print("size of cropped img: {}".format(img_crop.shape))
#    
#    new_size = (int(img_rot.shape[1]/2), int(img_rot.shape[0]/2))
#    img_rot_resized = cv2.resize(img_rot, new_size)
#    new_size = (int(img.shape[1]/2)), int(img.shape[0]/2)
#    img_resized = cv2.resize(img, new_size)
    
#    cv2.imshow("original contour", img_resized)
#    cv2.imshow("rotated image", img_rot_resized)
#    cv2.imshow("cropped_box", img_crop)
#    cv2.imwrite('line_parking_crop.png', img_crop)
    return img_crop;




def find(lpoints,x,y):
    temppoints=[]
    for p in lpoints:
        if(p[0]==x):
            temppoints.append(p)
        if(p[1]==y):
            temppoints.append(p)
    return temppoints
    

def sortPointAntiClockWise(lpoints,XXX,YYY):
    finalPoints=[]

    temppoints=find(lpoints,0,-1)
    if(len(temppoints)==1):
        finalPoints.append(temppoints[0])
    elif (len(temppoints)>1) :
        temppoints.sort()
        for i in range(1,-1,-1):
            finalPoints.append(temppoints[i])
    print(finalPoints)
    
    temppoints=find(lpoints, -1,0)
    if(len(temppoints)==1):
        finalPoints.append(temppoints[0])
    elif (len(temppoints)>1) :
        temppoints.sort()
        for i in range(2):
            finalPoints.append(temppoints[i])
            
    print(finalPoints) 
    
    temppoints=find(lpoints, XXX,-1)
    if(len(temppoints)==1):
        finalPoints.append(temppoints[0])
    elif (len(temppoints)>1) :
        temppoints.sort()
        for i in range(2):
            finalPoints.append(temppoints[i])
            
    print(finalPoints)  
            
    temppoints=find(lpoints, -1,YYY)
    if(len(temppoints) == 1):
        finalPoints.append(temppoints[0])
    elif (len(temppoints)>1) :
        temppoints.sort()
        for i in range(1,-1,-1):
            finalPoints.append(temppoints[i])
    print(finalPoints)  
    
 

    l=len(finalPoints)
    temp=finalPoints[l-1]
    i=0
    while(i<l):
        if((temp[0]==finalPoints[i][0])and (temp[1]==finalPoints[i][1])):
            finalPoints.remove(finalPoints[i])
            i-=1
            
        temp=finalPoints[i]
        i+=1
        l=len(finalPoints)
        

    return finalPoints
    


count=0
for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
        
        img = cv2.imread(inputpath+file)

    
           
#        img = cv2.imread(inputpath+'-141.jpg')
        xx,yy,zz=img.shape
        xxm=int(xx/2)
        yym=int(yy/2)
        
        #img = cv2.imread('parking.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),1)
        
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        #rotate = imutils.rotate_bound(edges, 90)
        
        
        #rotation angle in degree
        rotated = ndimage.rotate(edges, 90)
        
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 50  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 500  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on
        
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        print(lines)
        points = []
        line1=[]
        line2=[]
        line1Min=9999
        line2Min=9999
        slopeTotal=0
        
        if( (lines   is not None) and len(lines)>0):
        
            for line in lines:
                for x1, y1, x2, y2 in line:
                    points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            #        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    slopeTotal+=(y2-y1)/(x2-x1)
            #        print(slopeTotal)
                    print(xxm, yym)
                    print(x1, y1, x2, y2)
                    A,B,C=lineEquationA_B_C(x1,y1,x2,y2)
            #        print(A,B,C)
                    dis=distance(yym,xxm, A,B,C)
                    print("dis:"+str(dis))
                    if(y1<xxm):
                        if(dis<line1Min):
                            line1=((x1  , y1  ), (x2  , y2 ))
                    if(y1>xxm):
                        if(dis<line2Min):
                            line2=((x1 , y1  ), (x2  , y2 ))
                            
                    print("###########################")
                    
            print(line1)
            print(line2)
            if(len(line1)>0):
                print("")
            else:
                line1=((0 , 0  ), (0  , yy ))
               
            p1,p2 =line1
            cv2.line(line_image, p1, p2, (255, 255, 255), 1)
            
            l1points=getTwoIntersectionPoint(p1[0],p1[1],p2[0],p2[1],xx,yy)
                
     
                
            
            
            if(len(line2)>0):
                print("")
            else:
                line2=((xx , 0  ), (xx  , yy ))
                
            p1,p2 =line2
            cv2.line(line_image,p1, p2, (255, 255,255), 1)
            l2points=getTwoIntersectionPoint(p1[0]-1,p1[1],p2[0]-1,p2[1],xx,yy)
                
    
                
            lpoints=[]
            lpoints.append(l1points[0] )
            lpoints.append(l1points[1] )
            lpoints.append( l2points[0])
            lpoints.append( l2points[1])
            
            
            lpoints=sortPointAntiClockWise(lpoints,xx,yy) 
            
            crop_img = fourPointCrop(img,lpoints)
            #new_img = cv2.rectangle(img, l1points[0], l2points[1], (0, 255, 0), 5) 
            hh,ww,uu=crop_img.shape
            ww2= int(hh*2.08)
            www=int((ww-ww2)/2)
#            crop_img=crop_img[:,www:www+ww2]
             
             
            
    #        cv2.imwrite('line_parking_crop.png', new_img)
               
    
    
            file_output_path = os.path.join(outputpath+file+'-'+str(count)+'.jpg')
            directory = os.path.dirname(file_output_path)
            
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
            
            if(ww>(hh*1.5)):
                cv2.imwrite(file_output_path, crop_img)
            count+=1
    



 
#slope=slopeTotal/lines.shape[0]
#degree=( slope)*(180/3.1416)
#print(slope)
#print(degree)
#
#rotated = ndimage.rotate(img, degree)
#cv2.imwrite('line_parking_rotated.png', rotated)
#
#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
#print(lines_edges.shape)
#cv2.imwrite('line_parking.png', lines_edges)



#lines1 = cv2.HoughLinesP(rotated, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#print(lines1)
#points = []
#for line in lines1:
#    for x1, y1, x2, y2 in line:
#        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
#        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
#        slopeTotal+=(y2-y1)/(x2-x1)
#        print(slopeTotal)
#        
#slope=slopeTotal/lines.shape[0]
#degree=( slope)*(180/3.1416)
#print(slope)
#print(degree)
#
#lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
#print(lines_edges.shape)
#cv2.imwrite('line_parking1.png', lines_edges)





#print points
#intersections = bot.isect_segments(points)
#print intersections
#
#for inter in intersections:
#    a, b = inter
#    for i in range(3):
#        for j in range(3):
#            lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]
#
#cv2.imwrite('line_parking.png', lines_edges)


