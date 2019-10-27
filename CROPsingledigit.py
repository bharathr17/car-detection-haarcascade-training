import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
from scipy import ndimage
 

inputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/crop/number/"
outputpath="E:/PROJECT ALL/kaggle/project/dataExtract/VID7/crop/number/digits/"

#inputpath="D:/PROJECTS/Python/car video/found7/crop/number/"
#outputpath="D:/PROJECTS/Python/car video/found7/crop/number/digits/"

count=0
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
    
    
for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
        
        img = cv2.imread(inputpath+file)
#        img = cv2.imread('D:/PROJECTS/Python/car video/found7/crop/number/-913.jpg-256.jpg-174.jpg')
        

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),1)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        im = cv2.filter2D(blur_gray, -1, kernel)
#        im = cv2.GaussianBlur(im,(kernel_size, kernel_size),1)
        im = cv2.filter2D(im, -1, kernel)
        
#        low_threshold = 50
#        high_threshold = 150
#        edges = cv2.Canny(im, low_threshold, high_threshold)
#
#
##        img = cv2.imread('eGaIy.jpg', 0)
#        img1 = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)[1]  # ensure binary
#        img=img1
        ret, labels = cv2.connectedComponents(im)
        

        
        crop_img=imshow_components(labels)
        
        
#        blur_radius = 1.0
#        threshold = 50
#        
#        # smooth the image (to remove small objects)
#        imgf = ndimage.gaussian_filter(crop_img[:,:,1], blur_radius)
#        threshold = 50
#        
#        # find connected components
#        labeled, nr_objects = ndimage.label(imgf > threshold) 
#        print("Number of objects is {}".format(nr_objects))
#                
        
#        img = cv2.imread('ba3g0.jpg')
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        ret,thresh = cv2.threshold(crop_img[:,:,1],127,255,1)
#        contours,h = cv2.findContours(thresh,1,2)
#        for cnt in contours:
#          cv2.drawContours(img,[cnt],0,(0,0,255),1)
                
        # load image
#        img = cv2.imread('Image.jpg')
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        # threshold to get just the signature (INVERTED)
        retval, thresh_gray = cv2.threshold(crop_img[:,:,1],thresh=100,maxval=255,type=cv2.THRESH_BINARY_INV)
        
        image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
 
        c=0
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            area = w*h

            if ((area > (8*8))and (area < (20*20))):
                roi=crop_img[y:y+h,x:x+w,1]
                cv2.imwrite('Image_crop'+str(c)+'.jpg', roi)
                
                roi = image_resize(roi, height = 90)
                roi=fixedSize(roi)
                roi = image_resize(roi, height = 28)
                
                kernel_size = 5
                blur_gray = cv2.GaussianBlur(roi,(kernel_size, kernel_size),1)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                roi = cv2.filter2D(blur_gray, -1, kernel)
                
                
               
                
                file_output_path = os.path.join(outputpath+file+'-'+str(count)+"-"+str(c)+'.jpg')
                directory = os.path.dirname(file_output_path)
                
                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)
                
                if(roi.shape[0]==28 and roi.shape[1]==28 ):
                    cv2.imwrite(file_output_path, roi)
                count+=1
                
                
                
 
            c+=1
 



    
        # Output to files
#        roi=img[y:y+h,x:x+w]
#        cv2.imwrite('Image_crop.jpg', roi)
#        
#        cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
#         
#        cv2.imwrite("crop_single_digit.jpg", crop_img)
#        crop_img=img[0:,0:]
        
        
        
#        file_output_path = os.path.join(outputpath+file+'-'+str(count)+'.jpg')
#        directory = os.path.dirname(file_output_path)
#        
#        try:
#            os.stat(directory)
#        except:
#            os.mkdir(directory)
#        
#        
#        cv2.imwrite(file_output_path, crop_img)
#        count+=1




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



    