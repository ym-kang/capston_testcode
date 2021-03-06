from ctypes import *
import math
import random
import cv2
import numpy as np
import os 
from multiprocessing.pool import ThreadPool
from multiprocessing.sharedctypes import Value

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    new_values = values.ctypes.data_as(POINTER(ctype))
    return new_values

def array_to_image(arr):
    
    # need to return old values to avoid python freeing memory
    arr = cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr



import Queue
img_q = Queue.Queue(2)

def array_to_image_loop(get_frm_function):
    while True:
        frm = get_frm_function()
        if type(frm) is not np.ndarray:
            continue
        img = array_to_image(frm)
        if not img_q.full():
            img_q.put(img)


import threading
def runArrToImgLoop(get_frm_function):
    t = threading.Thread(target=array_to_image_loop,args=(get_frm_function,))
    t.start()


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#young-muk test
'''
lib_kym  = CDLL("libkym.so",RTLD_GLOBAL)    
ipl_arr_to_image = lib_kym.ipl_arr_to_image
ipl_arr_to_image.argtypes=[c_int,c_int,c_int,POINTER(c_float)]
ipl_arr_to_image.restype=IMAGE
'''
'''
getMatArr = lib_kym.getMatArr
getMatArr.restype = POINTER(c_float)

'''

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)

#lib = CDLL("./libdarknet.so")

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    

#import libkym
#from examples - detector-scipy-opencv.py
'''
def array_to_image(arr):

    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    
    arr = (arr/255.0).flatten()
    
    cur = time.time()
    data = c_array(c_float, arr)
    print 'elapsed:', time.time()-cur
    im = IMAGE(w,h,c,data)
    return im'''


def kym_arr_to_image(np_arr):
    c = np_arr.shape[2]
    h = np_arr.shape[0]
    w = np_arr.shape[1]
    np_arr = (np_arr).flatten()
    data = c_array(c_float, np_arr)
    im = ipl_arr_to_image(w,h,c,data)
    return im


#use ipl
def c_detect_cam(net, meta, c_img, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(cam_img, 0, 0)
    
    #im = kym_arr_to_image(cam_img)
    
    #rgbgr_image(c_img)
    
    
    im = c_img
    
import tool

def calcRange(j_range,meta,dets,res):
    for j in j_range:
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))


def detectFromQ(net, meta, thresh=.5, hier_thresh=.5, nms=.45):
    while img_q.empty():
        continue
    im,_ = img_q.get()
    
    num = c_int(0)
    pnum = pointer(num)
    tool.checkTime.printTime = False
    tool.checkTime('a')
    predict_image(net, im)
    tool.checkTime('b')
    #network_predict(net,im.data)
    tool.checkTime('c')
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    tool.checkTime('d')
    res = []

    #pool.apply_async(calc_result,args=(j,))

    #def calcrange(j_range):
    #for j in range(num):
    #    for i in range(meta.classes):
    #        if dets[j].prob[i] > 0:
    #            b = dets[j].bbox
    #            res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    threads = []
    for i in range(2):
        rng = range(num)[i::2]
        t = threading.Thread(target=calcRange,args=(rng,meta,dets,res))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    tool.checkTime('e')
    free_detections(dets, num)
    return res

def detect_numpy(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    
    im, arr = array_to_image(image)
    num = c_int(0)
    pnum = pointer(num)
    tool.checkTime.printTime = True
    tool.checkTime('a')
    predict_image(net, im)
    tool.checkTime('b')
    #network_predict(net,im.data)
    #tool.checkTime('c')
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    
    free_detections(dets, num)
    return res


import time
def detect_cam(net, meta, cam_img, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(cam_img, 0, 0)
    
    #im = kym_arr_to_image(cam_img)
    im,_ = array_to_image(cam_img)
    rgbgr_image(im)
    
    
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    
    free_detections(dets, num)
    return res

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    meta = load_meta("cfg/coco.data")
    import scipy.misc
    import time
    '''
    t_start = time.time()
    for ii in range(100):
        r = detect(net, meta, 'data/dog.jpg')
    print(time.time() - t_start)
    print(r)

    image = scipy.misc.imread('data/dog.jpg')
    for ii in range(100):
        scipy.misc.imsave('/tmp/image.jpg', image)
        r = detect(net, meta, '/tmp/image.jpg')
    print(time.time() - t_start)
    print(r)
    '''

    image = scipy.misc.imread('data/dog.jpg')
    t_start = time.time()
    for ii in range(100):
        r = detect_numpy(net, meta, image)
    print(time.time() - t_start)
    print(r)

