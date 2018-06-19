#-*- coding: utf-8 -*-
import numpy as np
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import math

def region_of_interest(img, vertices):

	mask = np.zeros_like(img)
	
	
	
	match_mask_color = 255 
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness = 3):
	
	if lines  is None:
		return

	img = np.copy(img)

	line_img = np.zeros(
		(
			img.shape[0],
			img.shape[1],
			3
		),
		dtype=np.uint8,
	)
	
	for line in lines:
		for x1, y1, x2, y2 in line:
			#cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

	#img = cv2.addWeighted(img, 1, line_img, 1, 0.0)
	
	return img
		



def detectLine(image):

	height = image.shape[0]
	width = image.shape[1]

	region_of_interest_vertices = [
		(width*0, height*4/5), #left low
		(width*2 / 5, height*1 / 3), #left high
		(width*3 / 5, height*1 / 3), #right high
		(width, height*4/5), #right low
	]
	#plt.figure()
	#plt.imshow(image)
	cv2.imshow('original',image)

	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	cannyed_image = cv2.Canny(gray_image, 100, 200)


	cropped_image = region_of_interest(
		cannyed_image,
			np.array([region_of_interest_vertices], np.int32),
	)
	cv2.imshow('crop',cropped_image)

	lines = cv2.HoughLinesP(
		cropped_image,
		rho = 6,
		theta=np.pi / 60,
		threshold=150,
		lines=np.array([]),
		minLineLength=40,
		maxLineGap=10
	)
	
	if lines is not None:
		leftLine,rightLine = polyfitLines(lines,image)
	else:
		leftLine = None
		rightLine = None


	
	return leftLine,rightLine

	#print("lines:",lines)

def polyfitLines(lines,image):
	left_line_x = []
	left_line_y = []
	right_line_x = []
	right_line_y = []

	for line in lines:
		for x1, y1, x2, y2 in line:
			if(x2-x1) is 0:
				continue
			slope = (y2 - y1) / (float)(x2 - x1)
			#print("slope",slope)
			if math.fabs(slope) < 0.5:
				continue
			if slope <= 0:
				left_line_x.extend([x1, x2])
				left_line_y.extend([y1, y2])
			else:
				right_line_x.extend([x1, x2])
				right_line_y.extend([y1, y2])


	min_y = image.shape[0] * (2.5 / 5.0)
	min_y = int(min_y)
	max_y = image.shape[0]


	

	leftLine = None
	rightLine = None
	
	if left_line_y != [] and left_line_y !=[]:
		poly_left = np.poly1d(np.polyfit(
			left_line_y,
			left_line_x,
			deg = 1
		))

		left_x_start = int(poly_left(max_y))
		left_x_end = int(poly_left(min_y))

		
		leftLine = {
			"x_start":left_x_start,
			"max_y":max_y,
			"x_end":left_x_end,
			"min_y":min_y
		}

	if right_line_y != [] and right_line_x !=[]:
		poly_right = np.poly1d(np.polyfit(
			right_line_y,
			right_line_x,
			deg = 1
		))

		right_x_start = int(poly_right(max_y))
		right_x_end = int(poly_right(min_y))

		
		rightLine = {
			"x_start":right_x_start,
			"max_y":max_y,
			"x_end":right_x_end,
			"min_y":min_y
		}

	'''
	line_image = draw_lines(
	image,
	[[ 
		[left_x_start, max_y, left_x_end, min_y],
		[right_x_start, max_y, right_x_end, min_y],
	]],
	thickness=5,
	)	
	'''
	return leftLine,rightLine

def draw_line(leftLine,rightLine,img):
	
	width = img.shape[1]
	height = img.shape[0]

	line_image = img

	if(leftLine!=None):
		if leftLine['x_start']>width*2/5:
			color = [0, 0, 255]
		else:
			color = [255, 0, 0]

		line_image = draw_lines(line_image,[[[leftLine['x_start'],leftLine['max_y'],leftLine['x_end'],leftLine['min_y']]]],color)
		

	if(rightLine!=None):
		if rightLine['x_start']<width*3/5:
			color = [0, 0, 255]
		else:
			color = [255, 0, 0]

		line_image = draw_lines(line_image,[[[rightLine['x_start'], rightLine['max_y'], rightLine['x_end'], rightLine['min_y']]]],color)
		

	
	return line_image




def testImage():
	#image = mpimg.imread('lane9.png')
	image = cv2.imread('lane7.png')

def testVideo():
	#vid = cv2.VideoCapture('videoplayback.mp4')
	vid = cv2.VideoCapture('../졸프영상파일/Test02.mp4')
	writer = cv2.VideoWriter()
	writer.open("out2.avi",cv2.VideoWriter_fourcc(*"H264"), 30, (1920,1080), True)


	while True:
		ret,im  = vid.read()
		if not ret:
			break

		leftLine,rightLine = detectLine(im)
		img = draw_line(leftLine,rightLine,im)
		cv2.imshow('line',img)
		cv2.waitKey(1)
		writer.write(img)
	
	writer.release()

def testOcam():
	import Stereo.liboCams as liboCams
	devpath = liboCams.FindCamera('oCam')
	if devpath is None:
		exit()

	test = liboCams.oCams(devpath, verbose=1)
	fmtlist = test.GetFormatList()
	ctrlist = test.GetControlList()
	test.Close()

	test = liboCams.oCams(devpath, verbose=0)
	test.Set(fmtlist[0])
	name = test.GetName()
	test.SetControl(10094850,400) # control exposure
	test.Start()

	while True:
		camL, camR = test.GetFrame(mode=2)
		rgbL = cv2.cvtColor(camL, cv2.COLOR_BAYER_GB2BGR)
		leftLine,rightLine = detectLine(rgbL)
		img = draw_line(leftLine,rightLine,rgbL)
		cv2.imshow('line',img)
		cv2.waitKey(1)
		print(leftLine,rightLine)


if __name__ == '__main__':
	testVideo()
	#testOcam()
	