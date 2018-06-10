import sys, os

import darknet as dn
import Location as loc
import time


loc.web_socket_server.RunServer() #run server socket server



from imutils.video import WebcamVideoStream
sys.path.append("../")
tiny = True
movie = True
dn.set_gpu(0)
if(not tiny):
    net = dn.load_net("cfg/yolov3.cfg","yolov3.weights",0)
    meta = dn.load_meta("cfg/coco.data")
elif(tiny):
    if(movie):
        #net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0)
        #meta = dn.load_meta("cfg/coco.data")
        #net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0)  mobilenet_yolo_coco
        net = dn.load_net("../dataset_test/cfg/yolov2-tiny-kym-run.cfg","backup1/yolov2-tiny-kym-train.backup",0)
        meta = dn.load_meta("../dataset_test/cfg/kym_test.data")
       
    else:
        net = dn.load_net("../test_data/test_training/yolov2-tiny-kym.cfg","backup/yolov2-tiny-kym_30000.weights",0)
        meta = dn.load_meta("../test_data/test_training/kym_test.data")
     
     #net = dn.load_net("cfg/yolov2-tiny-voc.cfg","backup/yolov2-tiny-voc.backup",0)
     #meta = dn.load_meta("cfg/voc.data")
else:
     net = dn.load_net("cfg/yolov3.cfg","yolov3.weights",0)
     meta = dn.load_meta("cfg/coco.data")
#kym-lib testing
'''
initializeCapture = dn.lib_kym.initializeCapture
initializeCapture.argtypes = [dn.c_char_p]

getImg = dn.lib_kym.getImg
#getImg.argtypes=[]
getImg.restype= dn.IMAGE

destroy_cap = dn.lib_kym.destroy_cap
videoplayback
initializeCam = dn.lib_kym.initializeCam
initializeCam.argtypes = [dn.c_int]True

show = dn.lib_kym.show
show.argtypes = [dn.IMAGE]'''


import numpy
'''
getCapMat = dn.lib_kym.getCapMat
getCapMat.restype = numpy.mat

getIpl = dn.lib_kym.getIpl
getIpl.restype = dn.c_void_p

toMat = dn.lib_kym.toMat

test = dn.lib_kym.test'''
#show.restype = 

#destroy.argtypes=[]
#destroy.restype = dn.c_void_p

import cv2
if(movie):
    #surf.ogv
    vid = cv2.VideoCapture("../dataset_test/data/test_movie/test_video_0510.mp4")
else:
    vid = cv2.VideoCapture(0)
#vs = WebcamVideoStream(src=0).start()

#cv2.VideoWriter_fourcc(*"XVID")
#vr = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
#initializeCapture("videoplayback.mp4")
#initializeCam(0)

#import libkym
#res = libkym.example_wrapper(([1],[2]))
#print dir(libkym)
#x=  libkym.cos_func(1)
#export PYTHONPATH=.
#import test_image
#import test_line
#based on image
def main2():
    for i,name in enumerate(test_image.imgs):
        path = test_image.base + name
        im = cv2.imread(path)
        start = time.time()
        r = dn.detect_numpy(net,meta,im)

      
        for i, k in enumerate(r):
            cv2.putText(im,k[0],(int(k[2][0]),int(k[2][1])),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),thickness=3)
            cv2.rectangle(im,
            (int(k[2][0]-k[2][2]/2),int(k[2][1]-k[2][3]/2)),
            (int(k[2][0]+k[2][2]/2),int(k[2][1]+k[2][3]/2))
            ,3)
        
        #show(im)

        #dn.free_image(im)
        #print r
        
        cv2.putText(im,name,(00,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow("img",im)
        cv2.waitKey(1)
        print "fps: ", 1/(time.time()-start)
        print "spf: ", (time.time()-start)
        1072

import Stereo
def main1():
    Stereo.main_stereo.RunThread() #read video, stereo calculation
    v = cv2.VideoWriter()
    #1280x720   1920x1072
    v.open("out.avi",cv2.VideoWriter_fourcc(*"H264"), 30, (1280,720), True)
    #cv2.
    frm=0
    while(True):


        start = time.time()
        #ret,im = vid.read()  
        if not Stereo.main_stereo.valueReady:
            continue  #value not ready -> wait


        im = Stereo.main_stereo.frameL
        
        #im = cv2.resize(im, (0,0), fx=0.5, fy=0.5) 

        r = dn.detect_numpy(net,meta,im,thresh=.5)
        
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
        #if(movie):
            #line = test_line.ImageView(img=im)
            #line.show(1)
        camdatas = []
        for i, k in enumerate(r):
            #if(k[0]!='jetson'):
            #    continue
            cv2.rectangle(im,
            (int(k[2][0]-k[2][2]/2),int(k[2][1]-k[2][3]/2)),
            (int(k[2][0]+k[2][2]/2),int(k[2][1]+k[2][3]/2))
            ,(0,0,255),thickness=2)
            cv2.putText(im,k[0],(int(k[2][0]),int(k[2][1])),cv2.FONT_HERSHEY_PLAIN,2,(120,120,32),thickness=2)
            cv2.putText(im,str(int(k[1]*100))+"%",(int(k[2][0]),int(k[2][1])+50),cv2.FONT_HERSHEY_PLAIN,2,(120,120,32),thickness=2)
            
            x = int(k[2][0])
            y = int(k[2][1])
            distance = 2
            name = k[0]

            camdatas.append(loc.sensor_data.CameraMath(x,y,distance,name))

        loc.sensor_data.cameraDatas = camdatas

        #show(im)
        
        #dn.free_image(im)
        #print r
        frm+=1
        #print("frm: ",frm)
        #if(frm%10==0):
        cv2.imshow("img",im)
        writeVideo = False
        if writeVideo:
            v.write(im)
        
        key = cv2.waitKey(1)
        if(key==ord('q')):
            break
        if(frm%100==0):
            print "fps: ", 1/(time.time()-start)
            print "spf: ", (time.time()-start)
        #if(frm>2400):
        #    break
            #pass
    v.release()

main1()


#main2()
       