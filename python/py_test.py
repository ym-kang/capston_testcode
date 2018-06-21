#-*- coding: utf-8 -*-
import sys, os
import darknet as dn
import Location as loc
import time
import numpy
import cv2
import math
import line_detection


sys.path.append("../")
tiny = True
movie = True
dn.set_gpu(0)

def loadnet():
    global net,meta

    if(not tiny):
        net = dn.load_net("cfg/yolov3.cfg","yolov3.weights",0)
        meta = dn.load_meta("cfg/coco.data")
    elif(tiny):
        if(movie):
            #net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0)
            #meta = dn.load_meta("cfg/coco.data")
            #net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0) # mobilenet_yolo_coco
            net = dn.load_net("../dataset_test/cfg/yolov2-tiny-kym-run.cfg","backup1/yolov2-tiny-kym-train.backup",0)
            meta = dn.load_meta("../dataset_test/cfg/kym_test.data")
            #net = dn.load_net("../dataset_test/cfg/yolov2-tiny-kym-run.cfg","backup1/yolov2-tiny-kym-train_470000.weights",0)
            
        else:
            net = dn.load_net("../test_data/test_training/yolov2-tiny-kym.cfg","backup/yolov2-tiny-kym_30000.weights",0)
            meta = dn.load_meta("../test_data/test_training/kym_test.data")
        
        #net = dn.load_net("cfg/yolov2-tiny-voc.cfg","backup/yolov2-tiny-voc.backup",0)
        #meta = dn.load_meta("cfg/voc.data")
    else:
        net = dn.load_net("cfg/yolov3.cfg","yolov3.weights",0)
        meta = dn.load_meta("cfg/coco.data")

def init():
    loc.web_socket_server.RunServer() #run server socket server
    loadnet()


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



#image + probablity, name, distance
def MarkImg(im, r, stereo=False):
    global Stereo
    camdatas = []
    for i, k in enumerate(r):
        #if(k[0]!='jetson'):
        #    continue
        x = int(k[2][0])
        y = int(k[2][1])
        if(stereo):
            distance,avg = Stereo.main_stereo.calculateDist(x,y)
        else:
            distance = 2

        if(distance>1):
            color = (120,120,32) #sky color
        else:
            color = (0,0,255) #red color

        cv2.rectangle(im,
        (int(k[2][0]-k[2][2]/2),int(k[2][1]-k[2][3]/2)),
        (int(k[2][0]+k[2][2]/2),int(k[2][1]+k[2][3]/2))
        ,color,thickness=2)
        cv2.putText(im,k[0],(int(k[2][0]),int(k[2][1])),cv2.FONT_HERSHEY_PLAIN,2,color,thickness=2)
        cv2.putText(im,str(int(k[1]*100))+"%",(int(k[2][0]),int(k[2][1])-50),cv2.FONT_HERSHEY_PLAIN,2,color,thickness=2)   
        #distance = Stereo.realsense.calculateDist(x,y)
        #distance = 2
        cv2.putText(im,str(round(distance,2) )+"m",(int(k[2][0]),int(k[2][1])+50),cv2.FONT_HERSHEY_PLAIN,2,color,thickness=2)
        name = k[0]
        camdatas.append(loc.sensor_data.CameraMath(x,y,distance,name))
    return camdatas

def MainRealSense():
    pass
    #Stereo.realsense.runRSCam()
    #if not Stereo.realsense.valueReady:
    #    continue
    #im = Stereo.realsense.img
    #im = vs.read()
    





threshold = .2


def MainOCAM():
    v = cv2.VideoWriter()
    v.open("out.avi",cv2.VideoWriter_fourcc(*"H264"), 10, (1280,960), True)
    global Stereo
    import Stereo
    import Stereo.main_stereo
    Stereo.main_stereo.showDepth=False
    Stereo.main_stereo.RunThread() #read video, stereo calculation

    frm=0

    while(True):
        if Stereo.main_stereo.stereoReady:
            break  #value not ready -> wait

    while(True):
        start = time.time()
        #ret,im = vid.read()  
        
        checkTime('initial')
        im = Stereo.main_stereo.frameL
        r = dn.detect_numpy(net,meta,im,thresh=threshold)     
        checkTime('detection')
        marked_im  = numpy.copy(im)
        camdatas = MarkImg(marked_im,r,stereo=True)
        loc.sensor_data.cameraDatas = camdatas

        line_left,line_right = line_detection.detectLine(im)
        marked_im = line_detection.draw_line(line_left,line_right,marked_im)
        checkTime('line detection')
        frm+=1
        cv2.imshow("img",marked_im)
        writeVideo = False
        if writeVideo:
            v.write(marked_im)
        key = cv2.waitKey(1)
        
        if(key==ord('q')):
            break
        if(frm%100==0):
            print "fps: ", 1/(time.time()-start)
            print "spf: ", (time.time()-start)
        

    if writeVideo:
        v.release()  
    sys.exit()

def checkTime(tag):
    if not hasattr(checkTime,'start'):
        checkTime.start = time.time()
    if not hasattr(checkTime,'printTime'):
        checkTime.printTime = False
    
    elapsed = time.time()-checkTime.start
    
    if checkTime.printTime:
        print(tag,"elapsed:",elapsed)
    checkTime.start = time.time()
    return elapsed



def MainVideo(video_name = "../dataset_test/data/test_movie/test_video_0531.mp4"):
    v = cv2.VideoWriter()
    v.open("out.avi",cv2.VideoWriter_fourcc(*"H264"), 10, (1280,720), True)
    vid = cv2.VideoCapture(video_name)
    frm=0
    
    while(True):
        start = time.time()
        ret, im = vid.read() 
        if not ret:
            break
        checkTime('initial')
        r = dn.detect_numpy(net,meta,im,threshold)     
        checkTime('detection')
        marked_im  = numpy.copy(im)
        camdatas = MarkImg(marked_im,r,stereo=False)
        loc.sensor_data.cameraDatas = camdatas

        line_left,line_right = line_detection.detectLine(im)
        marked_im = line_detection.draw_line(line_left,line_right,marked_im)
        checkTime('line detection')
        frm+=1
        cv2.imshow("img",marked_im)
        writeVideo = False
        if writeVideo:
            v.write(marked_im)
        key = cv2.waitKey(1)
        if(key==ord('q')):
            break
        if(frm%100==0):
            print "fps: ", 1/(time.time()-start)
            print "spf: ", (time.time()-start)

    if writeVideo:
        v.release()  
    sys.exit()
        

if __name__ is "__main__":
    init()
    #MainVideo('../dataset_test/data/test_movie/test_video_0619.mp4')
    #MainVideo('../졸프영상파일/Test01.mp4')
    MainOCAM()

#main2()
       