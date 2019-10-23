# -*- coding: utf-8 -*-

import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
 
import numpy as np

print(cv2.__version__)


 
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)





inputimagepath = 'E:/PROJECT ALL/kaggle/project/found2/found-1728.jpg'
inputimagepath = 'E:/PROJECT ALL/kaggle/project/found2/found-569.jpg'
outputpath = 'E:/PROJECT ALL/kaggle/project/datasetoutput/'


# Read the image
img = cv2.imread(inputimagepath, 0)
 
# Thresholding the image
(thresh, img_bin) = cv2.threshold(img, 0, 50,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)
# Invert the image
img_bin = 255-img_bin 
cv2.imwrite("Image_bin.jpg",img_bin)





# Defining a kernel length
kernel_length = np.array(img).shape[1]//80
 
# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))





# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)







# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_final_bin.jpg",img_final_bin) 




# Find contours for image, which will detect all the boxes
im2, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")









idx = 0
for c in contours:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img[y:y+h, x:x+w]
        cv2.imwrite( str(idx) + '.png', new_img)
# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img[y:y+h, x:x+w]
        cv2.imwrite( str(idx) + '.png', new_img)







# import the necessary packages




# load and show an image with Pillow
img = cv2.imread(inputimagepath)  
  
# Output img with window name as 'image' 
cv2.imshow('image', img)  
cv2.waitKey(0)         
cv2.destroyAllWindows() 

 
from PIL import Image
 
image = Image.open(inputimagepath)

for file in os.listdir(inputpath):
    if file.endswith(".xml"):
        print(file)
        fileName=file.split(".xml")[0]
        root = ET.parse(inputpath+'/'+file).getroot()
        realImagePath=item=root.find('path').text
        
        realimg = cv2.imread(realImagePath)
        
        count=0;
        for Variable in root.findall('object'):
            item=str(Variable.find('name').text)
            x1=int(Variable.find('bndbox/xmin').text)
            x2=int(Variable.find('bndbox/xmax').text)
            y1=int(Variable.find('bndbox/ymin').text)
            y2=int(Variable.find('bndbox/ymax').text)
            
            
            crop_img = realimg[y1:y2, x1:x2]
#            cv2.imshow("cropped", crop_img)
            file_output_path = os.path.join(outputpath+item+"/" , item+'-'+fileName+"-"+str(count)+'.jpg')
            directory = os.path.dirname(file_output_path)
            
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)
            
            
            cv2.imwrite(file_output_path, crop_img)
            count+=1
#            print(Variable.get('name'), Variable.text)
        
        
        
        
#        xmlPath=inputpath+'/'+file
#        print(xmlPath)
#        xmldoc = minidom.parse(xmlPath)
#        itemlist = xmldoc.getElementsByTagName('object')
#        print(len(itemlist))
#        print(itemlist[0].attributes['name'].value)
        
        
#        for s in itemlist:
#            print(s.attributes['name'].value)
        
#        print(os.path.join("/mydir", file))


#count=0
#while True:
#    ret, img = cap.read()
#    ret, img = cap.read()
#    if (type(img) == type(None)):
#        break
#    
#    cv2.imwrite(os.path.join(outputpath , 'Car-8-'+str(count)+'.jpg'), img)
##    cv2.imwrite("frame%d.jpg" % count, image)
#    count+=1
#    
#   

cv2.destroyAllWindows()