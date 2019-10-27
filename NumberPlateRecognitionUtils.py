import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os




class NumberPlateRecognitionUtils:
    
    def __init__(self):
        print("constractor NumberPlateRecognitionUtils")
        
        
    
    def imgToText(im):
        
    #    im = im.filter(ImageFilter.MedianFilter())
    #    enhancer = ImageEnhance.Contrast(im)
    #    im = enhancer.enhance(10)
    #    #im = im.convert('1')
    #    im.save("english_pp.jpg")
        
        pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
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



    def banglaTextToImage(d,file):
        base_size= 800,500
        base=np.zeros(base_size,dtype=np.uint8)
        base[:]=255
        
        pil_im = Image.fromarray(base)  
        draw = ImageDraw.Draw(pil_im)  
    #    font = ImageFont.truetype("Bangla_arial.ttf", 24) 
    #    font = ImageFont.truetype("Siyamrupali.ttf", 24) 
        font = ImageFont.truetype("Nikosh.ttf", 20) 
        draw.text( (0,0 ), d, font=font) 
    #    plt.savefig(outputpath+file+".jpg.jpg" )
        cv2.imwrite(outputpath+file+".jpg.jpg", np.array(pil_im)) 

        
    def show(img):
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    def imshow_components(labels):
        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        show(label_hue)
        blank_ch = 255*np.ones_like(label_hue)
    #    show(blank_ch)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    #    show(labeled_img)
    
        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
        # set bg label to black
        labeled_img[label_hue==0] = 0 
        show(labeled_img)
    
    #    cv2.imshow('labeled.png', labeled_img)
    #    cv2.waitKey()
        return labeled_img
   
    
    def fixedSize(img):
        
        h,w=img.shape[0:2]
        if(h<180 and w<180):
            
            ww=int((180-w)/2)
            
            base_size=180,180 
            base=np.zeros(base_size,dtype=np.uint8)
            base[:]=255
            base[45:h+45,ww:w+ww ]=img # this works
            plt.imshow(base, cmap='gray') 
            return base  
        else:
            return img
        
        
        
    def cropNumberPlate(img):
        edges = cv.Canny(img,100,200)
 
#        edges1 = cv.Canny(edges,100,200)
        x,y=edges.shape
        if(y>100):
         
            totalrow = [0]*(x)
            totalclm = [0]*(y)
            for row in range(x):
                for clm in range(y):
                    print(edges[row,clm])
                    totalrow[row]+=edges[row,clm]
                    totalclm[clm]+=edges[row,clm]
     
             
            
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
            
            return crop_img
        
        
        
        
    def cropNumberPlateArea(img):
        edges = cv.Canny(img,100,200)
            
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
                
   
        print( str( x1Index)+", " )
       
        
        crop_img=img
        
        retun crop_img
        
            
            
            
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
        
 
        
            
       
        
    
    
    
    
    
    
    
    
    
    def cropImageUsingVerticalLine(img):
        
        
           
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
            crop_img=crop_img[:,www:www+ww2]
            img=crop_img
            
        return img
    
    
    
    
    
             
             
    
    
 