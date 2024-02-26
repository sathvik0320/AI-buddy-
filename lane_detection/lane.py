#import pyrealsense2 as rs 
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as  mat
import matplotlib 
matplotlib.use('GTK3Agg')

#pipe = rs.pipeline()
#cfg = rs.config()

#cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
#cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)

#pipe.start(cfg)

#fits have corrdinates of there respective lines rght an left 
def average_slope(copy_image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_cordinates(copy_image, left_fit_average) if left_fit else np.array([0, 0])
    right_line = make_cordinates(copy_image, right_fit_average) if right_fit else np.array([0, 0])

    if np.all(right_line == 0):
        # Use coordinates of the other line
        return np.array([left_line])
    elif np.all(left_line == 0):
        # Use coordinates of the other line
        return np.array([right_line])
    else:
        # Both lines have non-zero coordinates
        return np.array([left_line, right_line])

def make_cordinates(copy_image, line_parameters):


    slope, intercept = line_parameters
    y1 = copy_image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

#reads image numpy array (imread) 
image = cv2.imread("lane.jpeg")
print(image)
#copyig image (duplcating the image)
copy_image = np.copy(image)

#while True :

# frame = pipe.wait_for_frames()
# color_frame = frame.get_depth_frame()
# gray scaling(step 1) 
gray_image = cv2.cvtColor(copy_image,cv2.COLOR_RGB2GRAY)
#reduce noise (gaussian blur ) 
blur = cv2.GaussianBlur(gray_image,(5,5),0)

#using canny for edga detections of lane with intensity chnage ascally doing derivative (change) in pixel values df(x,y)/dx
edges = cv2.Canny(blur,50,150)

#creating area of interest for image and cutting the area of interest from edges or canny images 
#in triangle chnage the number of cordinates respective to area of interest in you image 
triangle = np.array([(160,190),(1,156),(115,99)])
tri = np.zeros_like(edges)
cv2.fillPoly(tri,[triangle],255)
area = cv2.bitwise_and(tri,edges)
#lines in area of interest
lines = cv2.HoughLinesP(area,2,np.pi/180,100,np.array([]),minLineLength=0.5,maxLineGap=1)
#average slope of lines in edges 
average_lines = average_slope(copy_image,lines)
line_image = np.zeros_like(copy_image)
if average_lines is not None :
  for x1,y1,x2,y2 in average_lines: 
     cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)

#combining the orignal image and lines 
combine = cv2.addWeighted(copy_image,0.8,line_image,1,1)
#cv2.imshow("edges",edges)
#hough transformations 
plt.imshow(combine)
#cv2.waitKey(0)
plt.show()

#if we use video 
#video = cv2.VideoCapture("") 
#while(video.isOpened()):
# _,frame = video.read()

