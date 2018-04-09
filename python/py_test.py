import sys, os
sys.path.append("../")
import darknet as dn
import cv2
import time

from imutils.video import WebcamVideoStream


dn.set_gpu(0)
net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0)
meta = dn.load_meta("cfg/coco.data")

#kym-lib testing
initializeCapture = dn.lib_kym.initializeCapture
initializeCapture.argtypes = [dn.c_char_p]

getImg = dn.lib_kym.getImg
#getImg.argtypes=[]
getImg.restype= dn.IMAGE

destroy_cap = dn.lib_kym.destroy_cap

initializeCam = dn.lib_kym.initializeCam
initializeCam.argtypes = [dn.c_int]

show = dn.lib_kym.show
show.argtypes = [dn.IMAGE]


import numpy
getCapMat = dn.lib_kym.getCapMat
getCapMat.restype = numpy.mat

getIpl = dn.lib_kym.getIpl
getIpl.restype = dn.c_void_p

toMat = dn.lib_kym.toMat

test = dn.lib_kym.test
#show.restype = 

#destroy.argtypes=[]
#destroy.restype = dn.c_void_p

 

#cap = cv2.VideoCapture(0)
#vs = WebcamVideoStream(src=0).start()
vid = cv2.VideoCapture("videoplayback.mp4")
#cv2.VideoWriter_fourcc(*"XVID")
#vr = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
#initializeCapture("videoplayback.mp4")
#initializeCam(0)

#import libkym
#res = libkym.example_wrapper(([1],[2]))
#print dir(libkym)
#x=  libkym.cos_func(1)


while(True):
    start = time.time()
    ret,im = vid.read()
    r = dn.detect_cam(net,meta,im)
    
     #im = vs.read()
    #ret, im = vid.read()
    #im = getImg()
    '''
    mat = getCapMat()
    mat = numpy.asarray(mat)
    '''
    #test()
    #mat = getCapMat()
    #np_img = numpy.asanyarray(mat)

    #r = dn.c_detect_cam(net,meta,im)
    
    #continue
    
    
    for i, k in enumerate(r):
        cv2.putText(im,k[0],(int(k[2][0]),int(k[2][1])),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),thickness=3)
        cv2.rectangle(im,
         (int(k[2][0]-k[2][2]/2),int(k[2][1]-k[2][3]/2)),
        (int(k[2][0]+k[2][2]/2),int(k[2][1]+k[2][3]/2))
        ,3)
    
    #show(im)

    #dn.free_image(im)
    #print r
    cv2.imshow("img",im)
    cv2.waitKey(1)
    print "fps: ", 1/(time.time()-start)
    print "spf: ", (time.time()-start)
    


