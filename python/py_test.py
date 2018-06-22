#-*- coding: utf-8 -*-
import sys, os
import darknet as dn
import Location as loc
import time
import numpy
import cv2
import math
import line_detection
import tool
import threading
from multiprocessing.pool import ThreadPool

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
    runImgThread()
    dn.runArrToImgLoop(getFrm)
    v = cv2.VideoWriter()
    v.open("out.avi",cv2.VideoWriter_fourcc(*"H264"), 10, (1280,960), True)
    global Stereo,marked_im,frm_ready,cam_im
    import Stereo
    import Stereo.main_stereo
    Stereo.main_stereo.showDepth=False
    Stereo.main_stereo.RunThread() #read video, stereo calculation
    pool = ThreadPool(processes=2)
    frm=0

    while(True):
        if Stereo.main_stereo.stereoReady:
            break  #value not ready -> wait

    while(True):
        start = time.time()
        #ret,im = vid.read()  
        
        tool.checkTime('initial')
        im = Stereo.main_stereo.frameL
        cam_im = im
        #r = dn.detect_numpy(net,meta,im,thresh=threshold)     

        #r = dn.detectFromQ(net,meta,threshold)
        result_r = pool.apply_async(dn.detectFromQ,(net,meta,threshold))
        result_l = pool.apply_async(line_detection.detectLine,(im,))
        

        r = result_r.get()
        #tool.checkTime('detection1')
        line_left,line_right = result_l.get()
        #tool.checkTime('detection2')
        
        #r= t1.join()
        
        
        tool.checkTime('detection')
        marked_im_tmp  = numpy.copy(im)
        camdatas = MarkImg(marked_im_tmp,r,stereo=True)
        loc.sensor_data.cameraDatas = camdatas

        #line_left,line_right = line_detection.detectLine(im)
        marked_im = line_detection.draw_line(line_left,line_right,marked_im_tmp)
        frm_ready= True
        tool.checkTime('line detection')
        frm+=1
        #cv2.imshow("img",marked_im)
        writeVideo = False
        if writeVideo:
            v.write(marked_im)
        #key = cv2.waitKey(1)
        
        #if(key==ord('q')):
        #    break
        if(frm%100==0):
            print "fps: ", 1/(time.time()-start)
            print "spf: ", (time.time()-start)
        

    if writeVideo:
        v.release()  
    sys.exit()



def showImgLoop():
    global marked_im,frm_ready
    while True:
        if not 'marked_im' in globals():
            continue
        if not 'frm_ready' in globals():
            continue
        if not frm_ready:
            continue
        else:
            frm_ready = False
        cv2.imshow('output',marked_im)

        if line_detection.cropped_image is not None:
            cv2.imshow('cropped',line_detection.cropped_image)

        key = cv2.waitKey(1)

        if(key=='q'):
            sys.exit()

def runImgThread():
    import threading
    t = threading.Thread(target=showImgLoop)
    t.start()

def getFrm():
    if 'cam_im' not in globals():
        return None
    else:
        return cam_im

def MainVideo(video_name = "../dataset_test/data/test_movie/test_video_0531.mp4"):
    runImgThread()
    dn.runArrToImgLoop(getFrm)
    v = cv2.VideoWriter()
    v.open("out.avi",cv2.VideoWriter_fourcc(*"H264"), 10, (1280,720), True)
    vid = cv2.VideoCapture(video_name)
    frm=0
    pool = ThreadPool(processes=2)
    
    global marked_im,frm_ready,cam_im
    while(True):
        start = time.time()
        ret, im = vid.read() 
        cam_im = im
        if not ret:
            break
        tool.checkTime('initial')
        #r = dn.detect_numpy(net,meta,im,threshold)     
        #r = dn.detectFromQ(net,meta,threshold)

        
        result_r = pool.apply_async(dn.detectFromQ,(net,meta,threshold))
        result_l = pool.apply_async(line_detection.detectLine,(im,))
        

        r = result_r.get()
        #tool.checkTime('detection1')
        line_left,line_right = result_l.get()
        #tool.checkTime('detection2')

        
        marked_im_tmp  = numpy.copy(im)
        camdatas = MarkImg(marked_im_tmp,r,stereo=False)
        loc.sensor_data.cameraDatas = camdatas

        #line_left,line_right = line_detection.detectLine(im)
        marked_im = line_detection.draw_line(line_left,line_right,marked_im_tmp)
        frm_ready = True
        tool.checkTime('line detection')
        frm+=1
        #cv2.imshow("img",marked_im)
        writeVideo = False
        if writeVideo:
            v.write(marked_im)
        #key = cv2.waitKey(1)
        #if(key==ord('q')):
        #    break
        if(frm%100==0):
            print "fps: ", 1/(time.time()-start)
            print "spf: ", (time.time()-start)

    if writeVideo:
        v.release()  
    sys.exit()
        

if __name__ is "__main__":
    init()
    #MainVideo('videoplayback.mp4')
    #MainVideo('../dataset_test/data/test_movie/test_video_0619.mp4')
    #MainVideo('../졸프영상파일/Test01.mp4')
    MainOCAM()

#main2()
       