import sys, os
sys.path.append("../")
import darknet as dn
import cv2
import time


dn.set_gpu(0)
net = dn.load_net("cfg/yolov2-tiny.cfg","yolov2-tiny.weights",0)
meta = dn.load_meta("cfg/coco.data")

cap = cv2.VideoCapture(0)
while(True):
    start = time.time()
    ret,im = cap.read()
    r = dn.detect_cam(net,meta,im)
    for i, k in enumerate(r):
        cv2.rectangle(im,
         (int(k[2][0]-k[2][2]/2),int(k[2][1]-k[2][3]/2)),
        (int(k[2][0]+k[2][2]/2),int(k[2][1]+k[2][3]/2))
        ,3)
    #print r
    cv2.imshow("img",im)
    cv2.waitKey(1)
    print "fps: ", 1/(time.time()-start)
    print "spf: ", (time.time()-start)
    


