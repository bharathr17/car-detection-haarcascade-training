import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os


#img = cv.imread('C:/Users/Raju/Desktop/15704dsf73130667.jpg',0)
#img = cv.imread('D:/PROJECTS/Python/ruff/-4.jpg',0)

#inputpath="D:/PROJECTS/Python/car-number-plate-detection-haarcascade-training/p/"
#inputpath="D:/PROJECTS/Python/ruff2/"
#outputpath="D:/PROJECTS/Python/ruff3/"

inputpath="E:/PROJECT ALL/kaggle/project/found7/crop/number/"
outputpath="E:/PROJECT ALL/kaggle/project/found7/crop/number/digit/"

count=0

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

#    cv2.imshow('labeled.png', labeled_img)
#    cv2.waitKey()
    return labeled_img
    
    
    
for file in os.listdir(inputpath):
    if file.endswith(".jpg"):
        print(file)
        
        
        
        img = cv2.imread(inputpath+file,0)
#        img = cv2.imread('E:/PROJECT ALL/kaggle/project/found7/crop/number/-588.jpg-378.jpg-279.jpg',0)
        


#        img = cv2.imread('eGaIy.jpg', 0)
        img1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        img=img1
        ret, labels = cv2.connectedComponents(img)
        

        
        crop_img=imshow_components(labels)
#        crop_img=img[0:,0:]
        
        
        
        file_output_path = os.path.join(outputpath+file+'-'+str(count)+'.jpg')
        directory = os.path.dirname(file_output_path)
        
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        
        
        cv2.imwrite(file_output_path, crop_img)
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