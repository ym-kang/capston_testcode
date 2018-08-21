import numpy as np
import cv2
import liboCams
import time
import sys
from openpyxl import Workbook
from sklearn.preprocessing import normalize

#Filtering
kernel = np.ones((3,3),np.uint8)

'''
def coords_mouse_disp(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
	#print x,y,disp[y,x],filteredImg[y,x]
		average=0
		for u in range (-1,2):
			for v in range (-1,2):
				average += disp[y+u,x+v]
		average = average/9
		Distance = -593.93*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
		Distance = np.around(Distance*0.01, decimals=2)
		print('Distance : ' + str(Distance) + ' m')
'''
def coords_mouse_disp(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:	
		dist,disp = calculateDist(x,y)
		print("distance="+str(dist))
		print("disparity="+str(disp))

def calculateDist(x,y):
	average=0
	count = 0
	width = disp.shape[1]
	height = disp.shape[0]
	for u in range (-1,2):
		for v in range (-1,2):
			
			new_y = y+u
			new_x = x+v

			if new_x<0 or new_x>=width:
				continue
			if new_y<0 or new_y>=height:
				continue

			current_val =  disp[y+u,x+v]
			if(current_val<=0):
				continue
			count = count +1
			average += current_val
	if(count==0):
		average = 0
	else:
		average = average/float(count)

	#Distance = -593.93*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
	#ocam camera baseline(distance) : 12cm
	#sensor size 4.96x3.72mm  https://www.digicamdb.com/sensor-sizes/
	#focal length = 3.6mm
	Distance =0.0
	return map3d[y][x][2],average
	


	if(average!=0):
		Distance = 0.120 * 0.0036 /(average*0.00496/1280)
	
		#Distance = baseline * fical_length / (disparity * imagesansor_width/horizontal_resolution)
	#Distance = np.around(Distance*0.01, decimals=2)
	#print('x,y: '+str(x),str(y))
	#print('Distance : ' + str(Distance) + ' m')
	return Distance,average



wb = Workbook()
ws=wb.active

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

#Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# Prepare object points
objp = np.zeros((9*6,3), np.float32)
box_size =  24/1000.0 #checker board box size (m)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*box_size


def cali_save():
	# Arrays to store object points and image points from all images
	objpoints = []
	imgpointsR = []
	imgpointsL = []
	global retR, mtxR, distR, rvecsR, tvecsR, retS, MLS, dLS, MRS, dRS, R, T, E, F, RL, RR, PL, PR, Q, roiL, roiR, Shape
	# Start calibration from the camera
	print('Starting calibration for the 2 cameras... ')

	# Call all saved images
	for i in range (0,20):
		if(i%1!=0):
			continue
		t= str(i)
		print(t)
		ChessImgR = cv2.imread('python/Stereo/imgs/chessboard-R'+t+'.png',0)
		ChessImgL = cv2.imread('python/Stereo/imgs/chessboard-L'+t+'.png',0)
		ChessImgL = cv2.resize(ChessImgL,(0,0),fx=.5,fy=.5)
		ChessImgR = cv2.resize(ChessImgR,(0,0),fx=.5,fy=.5)
		
		retR, cornersR = cv2.findChessboardCorners(ChessImgR, (9,6),None)
		retL, cornersL = cv2.findChessboardCorners(ChessImgL, (9,6),None)

		if (retR == True) & (retL == True):
			objpoints.append(objp)
			cv2.cornerSubPix(ChessImgR, cornersR,(11,11),(-1,-1),criteria)
			cv2.cornerSubPix(ChessImgL, cornersL,(11,11),(-1,-1),criteria)
			imgpointsR.append(cornersR)
			imgpointsL.append(cornersL)

	Shape = ChessImgR.shape[::-1]
	# Determine the new values for different parameters
	#	 Right side
	retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImgR.shape[::-1],None,None)
	hR, wR = ChessImgR.shape[:2]
	OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR,hR),1,(wR,hR))

	#	Left side
	retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImgL.shape[::-1],None,None)
	hL, wL = ChessImgL.shape[:2]
	OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL,hL),1,(wL,hL))

	print('Cameras Ready to use')

	#********************************************
	#***** Calibrate the Cameras for Stereo *****
	#********************************************

	# StereoCalibrate function
	flags = 0
	flags |= cv2.CALIB_FIX_INTRINSIC
	#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
	#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
	#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
	#flags |= cv2.CALIB_FIX_ASPECT_RATIO
	#flags |= cv2.CALIB_ZERO_TANGENT_DIST
	#flags |= cv2.CALIB_RATIONAL_MODEL
	#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
	#flags |= cv2.CALIB_FIX_K3
	#flags |= cv2.CALIB_FIX_K4
	#flags |= cv2.CALIB_FIX_K5
	retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, ChessImgR.shape[::-1], criteria_stereo, flags)

	# StereoRectify function
	rectify_scale = 0
	RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, Shape, R, T, rectify_scale,(0,0)) # last parameter is alpha, if 0 croped, if 1 not croped

	fs = cv2.FileStorage()
	fs.open("cali.yaml",cv2.FILE_STORAGE_WRITE)
	fs.write("MLS",MLS)
	fs.write("dLS",dLS)
	fs.write("MRS",MRS)
	fs.write("dRS",dRS)
	fs.write("mtxR",mtxR)
	fs.write("mtxL",mtxL)
	fs.write("distR",distR)
	fs.write("distL",distL)
	fs.write("R",R)
	fs.write("T",T)
	fs.write("E",E)
	fs.write("F",F)

	#rectification
	fs.write("RL",RL)
	fs.write("RR",RR)
	fs.write("PL",PL)
	fs.write("PR",PR)
	fs.write("Q",Q)

	fs.write("Shape",Shape)
	

def cali_load():
	global retR, mtxR, distR, rvecsR, tvecsR, retS, MLS, dLS, MRS, dRS, R, T, E, F, RL, RR, PL, PR, Q, roiL, roiR, Shape
	fs = cv2.FileStorage()
	fs.open("cali.yaml",cv2.FILE_STORAGE_READ)
	MLS = fs.getNode("MLS").mat()
	dLS = fs.getNode("dLS").mat()
	MRS = fs.getNode("MRS").mat()
	dRS = fs.getNode("dRS").mat()
	mtxR = fs.getNode("mtxR").mat()
	mtxL = fs.getNode("mtxL").mat()
	mtxR = fs.getNode("distR").mat()
	mtxL = fs.getNode("distL").mat()


	RL = fs.getNode("RL").mat()
	RR = fs.getNode("RR").mat()
	PL = fs.getNode("PL").mat()
	PR = fs.getNode("PR").mat()
	Q = fs.getNode("Q").mat()
	R = fs.getNode("R").mat()
	T = fs.getNode("T").mat()
	E = fs.getNode("E").mat()
	F = fs.getNode("F").mat()
	Shape = fs.getNode("Shape").mat()

#cali_save()
cali_load()

# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, (Shape[0],Shape[1]), cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, (Shape[0],Shape[1]), cv2.CV_16SC2)

#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = -10
num_disp = 128  #-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
	numDisparities = num_disp,
	blockSize = 20,
	uniquenessRatio = 5,
	speckleWindowSize = 20,
	speckleRange = 7,
	disp12MaxDiff = 10,
	P1 = 8*3*window_size**2,
	P2 = 32*3*window_size**2,
	mode = cv2.StereoSGBM_MODE_SGBM)


# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS Filter Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#************************************
#***** Starting the StreoVision *****
#************************************
useOcam = False

if useOcam:
	# Call the two cameras
	devpath = liboCams.FindCamera('oCam')
	if devpath is None:
		exit()

	test = liboCams.oCams(devpath, verbose = 1)
	test.SetControl(10094850,400) # control exposure
	fmtlist = test.GetFormatList()
	ctrlist = test.GetControlList()
	test.Close()

	test = liboCams.oCams(devpath, verbose = 0)
	test.Set(fmtlist[0])
	name = test.GetName()
	test.Start()
else:
	cap = cv2.VideoCapture(0)

stereoReady=False
frameR = None
frameL = None
import time
def GetFrame():
	global disp,frameR,frameL,stereoReady
	count = 0
	print("GetFrame...")
	while True:
		count = count +1
		start = time.time()
		if useOcam:
			# start reading Camera images
			camR, camL = test.GetFrame(mode=2)

			tframeR = cv2.cvtColor(camR, cv2.COLOR_BAYER_GB2BGR)
			tframeL = cv2.cvtColor(camL, cv2.COLOR_BAYER_GB2BGR)

			frameR = cv2.resize(tframeR,(0,0),fx=0.5,fy=0.5)
			frameL = cv2.resize(tframeL,(0,0),fx=0.5,fy=0.5)

		else:
			ret,frm = cap.read()
			camR = camL = frm
			frameR = frameL = camR

		if(count>10):
			count = 0
			print("cam fps:"+str(round(1/(time.time()-start),2)))
		

def CalculateStereoTest(imgL, imgR):
	global disp,map3d
	Left_nice = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
	Right_nice = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
	if showDepth:
		cv2.imshow("Left_nice",Left_nice)
		cv2.imshow("Right_nice",Right_nice)
##	# Draw Red lines
##	For line in range(0, int(Right_nice.shape[0]/20)) : # Draw the lines on the images Then numer of line is defines by the image Size/20
##		Left_nice[line*20,:]=(0,0,255)
##		Right_nice[line*20,:]=(0,0,255)
##
##	For line in range(0, int(frameR.shape[0]/20)): # Draw the lines on the images then numer of line is defines by the image size/20
##		frameL[line*20,:]=(0,255,0)
##		frameR[line*20,:]=(0,255,0)

	# Show the Undistorted images
	#cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
	#cv2.imshow('Normal', np.hstack([frameL, frameR]))

	# Convert from color(BGR) to gray
	try:
		grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
		grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
	except:
		grayR = Right_nice
		grayL = Left_nice

	# Compute the 2 images for the Depth_image
	disp_temp = stereo.compute(grayL, grayR)#.astype(np.float32)/16
	
	dispL = disp_temp
	dispR = stereoR.compute(grayR, grayL)
	dispL = np.int16(dispL)
	dispR = np.int16(dispR)

	# Using the WLS filter
	filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
	filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
	filteredImg = np.uint8(filteredImg)
	#cv2.imshow('Disparity Map', filteredImg)
	disp=((disp_temp.astype(np.float32)/16.)-min_disp)#/num_disp
	
	#https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	map3d = cv2.reprojectImageTo3D(disp,Q)
	showDisparity(filteredImg)

	return Left_nice,Right_nice



showDepth= True
map3d=None
def GetFrameAndCalculateStereo():
	global disp,frameR,frameL,stereoReady,showDepth,map3d
	t = threading.Thread(target=GetFrame)
	t.start()

	count = 0
	print("running GetFrameAndCalculateStereo")
	while True:
		if(frameR is None):
			continue
		if(frameL is None):
			continue
		count = count+1
		start = time.time()
		
		# Rectify the images on rotation and alignement
		Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
		Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)
		if showDepth:
			cv2.imshow("Left_nice",Left_nice)
			cv2.imshow("Right_nice",Right_nice)
	##	# Draw Red lines
	##	For line in range(0, int(Right_nice.shape[0]/20)) : # Draw the lines on the images Then numer of line is defines by the image Size/20
	##		Left_nice[line*20,:]=(0,0,255)
	##		Right_nice[line*20,:]=(0,0,255)
	##
	##	For line in range(0, int(frameR.shape[0]/20)): # Draw the lines on the images then numer of line is defines by the image size/20
	##		frameL[line*20,:]=(0,255,0)
	##		frameR[line*20,:]=(0,255,0)


		# Convert from color(BGR) to gray
		grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
		grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

		# Compute the 2 images for the Depth_image
		disp_temp = stereo.compute(grayL, grayR)#.astype(np.float32)/16
		
		dispL = disp_temp
		dispR = stereoR.compute(grayR, grayL)
		dispL = np.int16(dispL)
		dispR = np.int16(dispR)

		# Using the WLS filter
		filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
		filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		filteredImg = np.uint8(filteredImg)
		#cv2.imshow('Disparity Map', filteredImg)
		disp=((disp_temp.astype(np.float32)/16)-min_disp)/num_disp

		#map3d= cv2.reprojectImageTo3D(disp,Q)

		if showDepth:
			showDisparity(filteredImg)
		stereoReady = True
	##	# Resize the image for faster executions
	##	dispR = cv2.resize(disp,None, fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

		if(count>1):
			count = 0
			print("depth fps:"+str(round(1/(time.time()-start),2)))
		

		

		# End the program
		#if cv2.waitKey(1) & 0xFF == ord(' '):
		#	break

	# Save excel
	##wb.save("data4.xlsx")

	# Release the Cameras
	test.Stop()
	cv2.destroyAllWindows()
	test.Close()
cv_window_wait_time = 1
def showDisparity(filteredImg):
	# Colors map
	# Filtering the Results with a closing filter
	closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
	dispc = (closing-closing.min())*255
	dispC = dispc.astype(np.uint8)
	disp_Color=cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
	filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

	# Show the result for the Depth_image
	#cv2.imshow('Disparity', disp)
	#cv2.imshow('Closing', closing)
	#cv2.imshow('Color Depth', disp_Color)
	cv2.imshow('Filtered Color Depth', filt_Color)

	# Mouse click
	cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filt_Color)
	cv2.waitKey(cv_window_wait_time)


import threading
def RunThread(): #run camera, stereo calculation
	t = threading.Thread(target=GetFrameAndCalculateStereo)
	t.start()
	return t


if __name__=="__main__":
	GetFrameAndCalculateStereo()