import numpy as np
import cv2
import liboCams
import time
import sys

print('Starting the Calibration just press the space bar to exit this part of the Program\n')
print('Push (s) to save the image you want and push (c) to see next frame without saving the image')

i=0
C=False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = []
imgpointsR = []
imgpointsL = []
useOcam = True

def resize(img):
	resized = cv2.resize(img,(0,0),fx=.7,fy=.7)
	return resized

if useOcam:
	# setting camera
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
else: 
	cap = cv2.VideoCapture(0)


# call the two camera
while True:
	if useOcam:
		camL, camR = test.GetFrame(mode=2)
	else:
		ret, camL = cap.read()
		camR= camL
		
	if useOcam:
		frameR = cv2.cvtColor(camR, cv2.COLOR_BAYER_GB2BGR)
		frameL = cv2.cvtColor(camL, cv2.COLOR_BAYER_GB2BGR)
		grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
		grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

	else:
		frameR = camR
		frameL = camL
		grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
		grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
	retL, cornersL = cv2.findChessboardCorners(grayL, (9,6), None)
	retR, cornersR = cv2.findChessboardCorners(grayR, (9,6), None)
	cv2.imshow('imgR', resize(frameR))
	cv2.imshow('imgL', resize(frameL))

# if found, add object points, image points (after refinding them)
	if (retL==True) & (retR==True) & (False == C):
		objpoints.append(objp)

# Refining the Position
		corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
		imgpointsR.append(cornersR)

		corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
		imgpointsL.append(cornersL)

# Draw and display the corners
		cv2.drawChessboardCorners(grayR,(9,6),corners2R,retR)
		cv2.drawChessboardCorners(grayL,(9,6),corners2L,retL)
		cv2.imshow('Frame R', resize(grayR))
		cv2.imshow('Frame L', resize(grayL))
		if cv2.waitKey(0) & 0xFF == ord('s'):
			t = str(i)
			print('Saved'+t)

			# Save the image
			cv2.imwrite('python/Stereo/imgs/chessboard-R'+t+'.png', frameR)
			cv2.imwrite('python/Stereo/imgs/chessboard-L'+t+'.png', frameL)
			i=i+1
		else:
			print('canceled')

	# End the program
	if cv2.waitKey(1) & 0xFF == ord(' '):
		break

test.Stop()
cv2.destroyAllWindows()
test.Close()
