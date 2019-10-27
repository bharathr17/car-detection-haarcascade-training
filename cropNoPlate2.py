import numpy as np
import cv2 as cv
import cv2 as cv2
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from PIL import ImageFont, ImageDraw 

 
#img = cv.imread('C:/Users/Raju/Desktop/15704dsf73130667.jpg',0)
inputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/"
outputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/test1/"
statistic = [0]*(10)
 
#inputpath="D:/PROJECTS/Python/car-number-plate-detection-haarcascade-training/p/"
#outputpath="D:/PROJECTS/Python/ruff/"
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


def banglaTOImage(d,file):
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


for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
##        Simple Thresholding
        img = cv.imread(inputpath+file,0)
        print(img.shape[1])
        if(img.shape[1]>100):
            img = image_resize(img, height = 100)
#            img = cv.imread(inputpath+"-23.jpg",1)
            
            
            
            ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
            ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
            ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
            ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
            
            titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
            images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
            s=""
            
            for i in  range(6):
                plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
                ss=imgToText(images[i]) 
                if(ss!=""):
                    statistic[i]+=1
                plt.title(titles[i]+ss)
                plt.xticks([]),plt.yticks([])
                s+=ss+"\n"+str(i)+"#"
                
    #        banglaTOImage(s,file)
    #        plt.show()
            plt.savefig(outputpath+file)
            plt.show()
            
            
            
            
            
            
    #  Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst      
    #        
    #dst – Destination image of the same size and the same type as src .
    #maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    #adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
    #thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    #blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
    #        Adaptive Thresholding
    #        img = cv2.imread(inputpath+"-23.jpg",0)
            img = cv2.imread(inputpath+file,0)
            img = image_resize(img, height = 100)
    #        img = cv2.medianBlur(img,5)
    
            ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,0)
            th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)
            
            titles = ['Original Image', 'Global Thresholding (v = 127)',
                        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
            images = [img, th1, th2, th3]
            
    #        titles.extend(['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV'])
    #        images.extend([img, thresh1, thresh2, thresh3, thresh4, thresh5])
    
    
    #        s=""
            for i in  range(4):
                plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #            plt.title(titles[i])
                ss=imgToText(images[i]) 
                if(ss!=""):
                    statistic[6+i]+=1
                plt.title(titles[i]+ss)
                plt.xticks([]),plt.yticks([])
                s+=ss+"\n"+str(6+i)+"#"
                
            banglaTOImage(s,file)
    #        plt.show()
            plt.savefig(outputpath+file +".jpg" )
            
            print(statistic)
            
            
    ##        Otsu’s Binarization
    #        
    #        img = cv2.imread(inputpath+"-23.jpg",0)
    #
    #        # global thresholding
    #        ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #        
    #        # Otsu's thresholding
    #        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #        
    #        # Otsu's thresholding after Gaussian filtering
    #        blur = cv2.GaussianBlur(img,(5,5),0)
    #        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #        
    #        # plot all the images and their histograms
    #        images = [img, 0, th1,
    #                  img, 0, th2,
    #                  blur, 0, th3]
    #        titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
    #                  'Original Noisy Image','Histogram',"Otsu's Thresholding",
    #                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    #        
    #        for i in  range(3):
    #            plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #            plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #            plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #            plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #            plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #            plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    #        plt.show()
    #                
    #        
    #        
    #        
    #        
    #        
    #        
    #        
    #        
    #         
    #        
    #        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #        plt.imshow(img,cmap = 'gray')
    #        
    #        mean, std = cv.meanStdDev(img)
    #        grayImg = (img -mean)/(1.e-6 + std)
    #        plt.imshow(grayImg,cmap = 'gray')
    #        
    #        
    #        
    #        kernel_size = 5
    #        blur_gray = cv.GaussianBlur(grayImg,(kernel_size, kernel_size),1)
    #        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #        sharpImg = cv.filter2D(blur_gray, -1, kernel)
    #        sharpImg = cv.filter2D(sharpImg, -1, kernel)
    #        
    #        
    # 
    #        
    #        
    #        edges = cv.Canny(img,100,200)
    # 
    ##        edges1 = cv.Canny(edges,100,200)
    #        x,y=edges.shape
    #        if(y>100):
    #         
    #            totalrow = [0]*(x)
    #            totalclm = [0]*(y)
    #            for row in range(x):
    #                for clm in range(y):
    #                    print(edges[row,clm])
    #                    totalrow[row]+=edges[row,clm]
    #                    totalclm[clm]+=edges[row,clm]
    #     
    #             
    #            
    #            mid=int(x/2)
    #            x1=mid-1
    #            x2=mid+1
    #            x1Max=-1
    #            x2Max=-1
    #            x1Index=0
    #            x2Index=0
    #            for row in range(mid):
    #                if(x1 >= 0):
    #                    if( totalrow[x1] > x1Max):
    #                        x1Max= totalrow[x1]
    #                        x1Index=x1
    #                if(x2 < x  ):
    #                    print(x2)
    #                    if( totalrow[x2] > x2Max):
    #                        x2Max= totalrow[x2]
    #                        x2Index=x2
    #                    
    #                x1=x1-1
    #                x2=x2+1
    #                
    #             
    #            
    #            
    #                
    #            
    #            mid=int(y/2)
    #            y1=mid-1
    #            y2=mid+1
    #            y1Max=-1
    #            y2Max=-1
    #            y1Index=0
    #            y2Index=0
    #            for row in range(mid):
    #                if(y1 >= 0):
    #                    if( totalclm[y1] > y1Max):
    #                        y1Max= totalclm[y1]
    #                        y1Index=y1
    #                if(y2 < y  ):
    #                    print(y2)
    #                    if( totalclm[y2] > y2Max):
    #                        y2Max= totalclm[y2]
    #                        y2Index=y2
    #                    
    #                y1=y1-1
    #                y2=y2+1
    #                
    #             
    #                
    #                
    #            print( str( x1Index)+", "+str(y1Index))
    #            print( str( x2Index)+", "+str(y2Index))
    #             
    #            
    #            crop_img=img[x1Index:x2Index,y1Index:y2Index]
    #            
    #            
                
    #            file_output_path = os.path.join(outputpath+'-'+str(count)+'.jpg')
    #            directory = os.path.dirname(file_output_path)
    #            
    #            try:
    #                os.stat(directory)
    #            except:
    #                os.mkdir(directory)
    #            
    #            
    #            cv.imwrite(file_output_path, crop_img)
    #            count+=1
    #    
    #    
    #
    #
    ##x1Index
    ##y1Index
    ##x2Index
    ##y2Index
    #    
    #    
    #    
    #
    # 
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    # 
