import cv2
import numpy as np
#import isect_segments_bentley_ottmann.poly_point_isect as bot

inputpath="E:/PROJECT ALL/kaggle/project/found7/"
outputpath="E:/PROJECT ALL/kaggle/project/found7/crop/"

img = cv2.imread(inputpath+'-307.jpg')
#img = cv2.imread('parking.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),1)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
#rotate = imutils.rotate_bound(edges, 90)
from scipy import ndimage

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
for line in lines:
    for x1, y1, x2, y2 in line:
        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
print(lines_edges.shape)
cv2.imwrite('line_parking.png', lines_edges)




#lines1 = cv2.HoughLinesP(rotated, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#print(lines1)
##points = []
#for line in lines1:
#    for x1, y1, x2, y2 in line:
#        points.append((( y1 + 0.0,x1 + 0.0), ( y2 + 0.0,x2 + 0.0)))
#        cv2.line(line_image, ( y1,x1), ( y2,x2), (255, 0, 0), 1)
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