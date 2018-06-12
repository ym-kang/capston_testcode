
import time
import numpy as np
import cv2
import pyrealsense as pyrs
from pyrealsense.constants import rs_option



def convert_z16_to_bgr(frame):
    '''Performs depth histogram normalization
    This raw Python implementation is slow. See here for a fast implementation using Cython:
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/cython_methods/methods.pyx
    '''
    hist = np.histogram(frame, bins=0x10000)[0]
    hist = np.cumsum(hist)
    hist -= hist[0]
    rgb_frame = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)

    zeros = frame == 0
    non_zeros = frame != 0

    f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
    rgb_frame[non_zeros, 0] = 255 - f
    rgb_frame[non_zeros, 1] = 0
    rgb_frame[non_zeros, 2] = f
    rgb_frame[zeros, 0] = 20
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 0
    return rgb_frame

def depthCam():
    depth_fps = 90
    depth_stream = pyrs.stream.DepthStream(fps=depth_fps)

    with pyrs.Service() as serv:
        with serv.Device(streams=(depth_stream,)) as dev:

            dev.apply_ivcam_preset(0)

            try:  # set custom gain/exposure values to obtain good depth image
                custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
                                (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
                dev.set_device_options(*zip(*custom_options))
            except pyrs.RealsenseError:
                pass  # options are not available on all devices

            cnt = 0
            last = time.time()
            smoothing = 0.9
            fps_smooth = depth_fps

            while True:

                cnt += 1
                if (cnt % 30) == 0:
                    now = time.time()
                    dt = now - last
                    fps = 30/dt
                    fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                    last = now

                dev.wait_for_frames()
                d = dev.depth
                d = convert_z16_to_bgr(d)

                cv2.putText(d, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

                cv2.imshow('', d)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

valueReady = False
def test1():
    global valueReady,img,depth
    import logging
    logging.basicConfig(level=logging.INFO)

    import pyrealsense as pyrs
    from pyrealsense.constants import rs_option

    with pyrs.Service() as serv:
        with serv.Device() as dev:

            dev.apply_ivcam_preset(0)

            try:  # set custom gain/exposure values to obtain good depth image
                custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 500.0),
                                (rs_option.RS_OPTION_R200_LR_GAIN, 500.0),
                                (rs_option.RS_OPTION_R200_DEPTH_CONTROL_NEIGHBOR_THRESHOLD,0),
                                (rs_option.RS_OPTION_R200_DEPTH_CLAMP_MIN,0),
                                #(rs_option.RS_OPTION_R200_DEPTH_CLAMP_MAX,99999)
                                ]
                dev.set_device_options(*zip(*custom_options))
            except pyrs.RealsenseError:
                pass  # options are not available on all devices

            cnt = 0
            last = time.time()
            smoothing = 0.9
            fps_smooth = 30

            while True:

                cnt += 1
                if (cnt % 10) == 0:
                    now = time.time()
                    dt = now - last
                    fps = 10/dt
                    fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                    last = now

                dev.wait_for_frames()
                
                c = dev.color
                c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
                img = c
                d = dev.depth * dev.depth_scale * 1000
                depth = d
                valueReady = True


                #print(d[240][320])
                d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_BONE)

                cd = np.concatenate((c, d), axis=1)

                cv2.putText(cd, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))

                cv2.imshow('', cd)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def test2():
    cap = cv2.VideoCapture(0)
    while True:
        ret,im = cap.read()
        #try:
        cv2.imshow("img",im)
        #except:
        #    pass
        cv2.waitKey(1)

def calculateDist(x,y):
    global depth
    average=0
    count = 0
    for u in range (-1,2):
        for v in range (-1,2):    
            cur_depth = depth[y+u,x+v]
            
            if(cur_depth is not 0):
                count = count +1
                average += depth[y+u,x+v]
   
    if count is not 0:
        average = average/count
    return average/1000.0

import threading
def runRSCam():
    #t1 = threading.Thread(target=depth)
    t2 = threading.Thread(target=test1)
    #t1.start()
    t2.start()
    #t1.join()
    #t2.join()

#test1()