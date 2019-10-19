# -*- coding: utf-8 -*-

import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


print(cv2.__version__)



inputpath = 'E:/PROJECT ALL/kaggle/project/car video/car image 8/'
outputpath = 'E:/PROJECT ALL/kaggle/project/datasetoutput/'


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