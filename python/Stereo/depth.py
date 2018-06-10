import numpy as np
import cv2
from matplotlib import pyplot as plt

def testImgFile():
    imgL=cv2.imread('ImageL.png',0)
    imgR=cv2.imread('ImageR.png',0)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity,"gray")
    plt.show()

#split array reference
#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.split.html
#calculate disparity
def calcDisparityCamImg(camImg):
    camImg = cv2.cvtColor(camImg,cv2.COLOR_BGR2GRAY)
    width = camImg.shape[1] #check width size
    imgL = camImg[:,0:int(width/2)] #split array (L and R)
    imgR = camImg[:,int(width/2):width]
    cv2.imshow('L',imgL)
    cv2.imshow('R',imgR)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity)
    plt.pause(0.01)
    return disparity
