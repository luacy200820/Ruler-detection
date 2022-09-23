import cv2
import numpy as np 
import glob
import matplotlib.pyplot as plt
import imutils
from scipy import signal

def find_contour(hsv,low,high,img,center):
    mask = cv2.inRange(hsv,low,high)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=4)
    plt.imshow(opening,cmap='gray')
    plt.show()
 
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    rect_img = img.copy()
    for c in cnts: 
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(rect_img,[box],0,(0,0,255),1)
        # plt.imshow(rect_img[:,:,(2,1,0)])
        # plt.show()
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
 
        center.append([cY,cX,box])
 
        
    return cnts,center


def sharpen(img, sigma=300):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def cropImage(point,image,position_blue):
    crop_image = image.copy()
    """Get width and height distances"""
    w = abs(point[0][2][0] - point[0][2][1])
    h = abs(point[0][2][1] - point[0][2][2])
    width_len = 0
    height_len = 0

    if w[0] != 0 and w[0] > w[1]:
        width_len = w[0]
    elif w[1] != 0:
        width_len = w[1]
    if h[0] != 0 and h[0] > h[1] :
        height_len = h[0]
    elif h[1] != 0:
        height_len = h[1]
        
 
    if position_blue == 'left':    
        crop=crop_image[point[1][:2][0]-int(height_len/2):point[1][:2][0]+int(width_len/2),point[1][:2][1]+int(height_len/2):point[1][:2][1]+int(1.5*height_len)]
        
    elif position_blue == 'down': #find right block
        crop=crop_image[point[1][:2][0]-int(height_len/2):point[1][:2][0]+int(height_len/2),point[1][:2][1]+int(height_len/2):point[1][:2][1]+int(height_len)+int(height_len/2)]
    
    elif position_blue == 'right':
        crop=crop_image[point[1][:2][0]-int(height_len/2):point[1][:2][0]+int(width_len/2),point[1][:2][1]-int(1.5*height_len):point[1][:2][1]-int(height_len/2)]
    
    elif position_blue == 'upper':
        crop=crop_image[point[1][:2][0]-int(height_len/2):point[1][:2][0]+int(height_len/2),point[1][:2][1]-int(height_len)-int(height_len/2):point[1][:2][1]-int(height_len/2)]

    sharp_crop = sharpen(crop)
    return sharp_crop

def getEdgeline(crop_img):
    """crop_img : crop image, laplacian binary, and threshold"""
    gray_crop = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    blurred_crop = cv2.bilateralFilter(gray_crop,5,75,75)

    # convolute with proper kernels
    laplacian_crop = cv2.Laplacian(blurred_crop,cv2.CV_64F)

    th,binary = cv2.threshold(laplacian_crop,5,255,cv2.THRESH_BINARY)
    binary = np.uint8(binary)

    return binary

def findLines(binary,crop):
    base = cv2.HoughLinesP(binary, 1, np.pi / 180,18, minLineLength=10, maxLineGap=12)
    pixel_array = []
    show_img = crop.copy()

    if base is not None:
        for line in base:
            x1, y1, x2, y2 = line[0]
            cv2.line(show_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            pixel_length = np.abs(x2 - x1)
            pixel_array.append([(x1, y1), (x2, y2)])
 
    return pixel_array

def findVertialLine(crop,line_array):
    """
    crop:crop image，vertical_array: line position
    return vertical_line: vertical line axis
    """
    vertical_line = [] 
    line_img = crop.copy()

    for i in range(len(line_array)):
        point1 = line_array[i][0]
        point2 = line_array[i][1]
        
        if point1[0] != point2[0]:
            slope = (point2[1] - point1[1]) / ( point1[0] - point2[0])
        else:
            slope = 0
        
        # Center point coordinates
        midpoint = (int((point1[0]+point2[0]) /2) ,int((point1[1]+point2[1]) /2))
        
        x1 = point1[0] -60 
        x2 = point1[0] +60
 
        y1 = int(slope * ( x1 - midpoint[0]) + midpoint[1] )
        y2 = int(slope * ( x2 - midpoint[0]) + midpoint[1] )
 
        vertical_line.append([slope,(x2,y2),(x1,y1),midpoint])
        cv2.line(line_img,(x2,y2),(x1,y1),(0,0,255),1)
 
    return vertical_line

def getFrequencyLine(vertical_array,crop):
    save_paraline = []
    for i in range(len(vertical_array)):
        s = vertical_array[i][0] # slope
        if s == 0 : #leaving a slope of 0
            save_paraline.append(vertical_array[i])

    line_save_posit = []
    temp = 255
    for line in range(len(save_paraline)):
        line_img2 = crop.copy()
        hsv_2 = crop.copy()
        hsv_2 = cv2.cvtColor(hsv_2,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv_2)
        #Sort the midpoint on the y-axis
        save_paraline.sort(key=lambda x:x[3][1])

        # Change negative image, find wave crest
        pixel_value = []
        test = crop.copy()
        gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
        negative = 255-gray #negative image
        W = crop.shape[1]
        line_ = save_paraline[line][1][1]
        
        if line_ > 3 and W-line_ >4:
            for w in range(W):
                if s[line_][w] < 80:
                    pixel_value.append(negative[line_][w])
    
            num_peak_3 = signal.find_peaks(pixel_value ,height=70, distance=2)
            print(num_peak_3[0])
            if len(num_peak_3[0]) < 3:
                print('not this line!!')
                continue
            elif len(num_peak_3[0]) >= 4:
                print('the number of peaks is ' + str(len(num_peak_3[0])))
                
                # Several crests represent several 0.1cm
                diffPixel = num_peak_3[0][-1] - num_peak_3[0][0]
                line_save_posit.append([line_,diffPixel,len(num_peak_3[0])-1,num_peak_3[0]])
                break
    return line_save_posit

def calculatePix(objectMask,pixel):
    w,h = objectMask.shape[:2]
    num = 0 #pixel total number
    for i in range(w):
        for j in range(h):
            if objectMask[i][j] == pixel:
                num +=1
    return num

def getMaskBoundary(wound,msk_path): 
    wound_image = wound.copy()
    gray2 = cv2.cvtColor(wound_image,cv2.COLOR_BGR2GRAY)
    
    objectPixel = calculatePix(gray2,255)
    return objectPixel
    
def calculateArea(line_array,objectPixel):
    
    ratio_pixel_cm = line_array[0][1]* line_array[0][1]
    real_area = round(int(line_array[0][2])/10 * int(line_array[0][2])/10,3)
    area = objectPixel * real_area / ratio_pixel_cm

    return area

if __name__ == "__main__":
 
    img_path = 'image'
    msk_path = 'label'
    image =cv2.imread(img_path)
    image = cv2.resize(image,(600,600))
    wound = cv2.imread(msk_path,1)
    wound = cv2.resize(wound,(600,600))
    img_ = image.copy()
    crop = img_[390:440,310:360] #Find your own place 
    
    binarzie = getEdgeline(crop) 
    line_array = findLines(binarzie,crop)
    vertical_array = findVertialLine(crop,line_array)
    point_posit = getFrequencyLine(vertical_array,crop)
    objectPixel = getMaskBoundary(wound,msk_path)
    realArea = calculateArea(point_posit,objectPixel)
    
    print("Final area",realArea)