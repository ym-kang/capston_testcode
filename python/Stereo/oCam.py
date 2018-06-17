# /usr/bin/env python2

import liboCams
import cv2
import time
import sys
import numpy as np
#from matplotlib import pyplot as plt

devpath = liboCams.FindCamera('oCam')
if devpath is None:
	exit()

test = liboCams.oCams(devpath, verbose=1)
fmtlist = test.GetFormatList()
ctrlist = test.GetControlList()
test.SetControl(10094850,400) # control exposure
test.Close()

test = liboCams.oCams(devpath, verbose=0)
test.Set(fmtlist[0])
name = test.GetName()
test.Start()

start_time = time.time()

frame_cnt = 0

while True:
	
	
	stereo = cv2.StereoBM_create(numDisparities = 16 , blockSize = 15)
	frameL, frameR = test.GetFrame(mode=2)
	
	grayR = cv2.cvtColor(frameR, cv2.COLOR_BAYER_GB2GRAY)
	grayL = cv2.cvtColor(frameL, cv2.COLOR_BAYER_GB2GRAY)

	rgbR = cv2.cvtColor(frameR, cv2.COLOR_BAYER_GB2BGR)
	rgbL = cv2.cvtColor(frameL, cv2.COLOR_BAYER_GB2BGR)

	disparity = stereo.compute(grayL, grayR)

#	plt.imshow(disparity,'gray')
#	plt.ion()
#	plt.show()
#	plt.pause(0.00000001)

	cv2.imshow(test.cam.card+' L', rgbL)
	cv2.imshow(test.cam.card+' R', rgbR)
	cv2.imshow('disparity', disparity)

	char = cv2.waitKey(1)
	if char == 27:
#		cv2.imwrite('disparity.png', disparity)
		cv2.imwrite('left002.png', rgbL)
		cv2.imwrite('right002.png', rgbR)
		break
	frame_cnt += 1

print ('Result Frame Per Second : ', frame_cnt/(time.time()-start_time))
test.Stop()
cv2.destroyAllWindows()
char = cv2.waitKey(1)
test.Close()
