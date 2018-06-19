# -*- coding: UTF-8
from collections import namedtuple
import cv2
import numpy as np
#import matplotlib.pyplot as plt


class LaneInfo:
    xpos=0
    ypos=0
    LaneIndex=0
    @staticmethod
    def filterLoad(img,gray,threshold = 240):
        low_y  = np.array([20,100,100], dtype = 'uint8') #노란색 추출 시작값
        upp_y = np.array([30,255,255], dtype = 'uint8')  #노란색 추출 끝값
        mask_y = cv2.inRange(img,low_y,upp_y) #노란색 차선 추출
        mask_w = cv2.inRange(gray,threshold,255) #일반 차선 추출
        mask_yw = cv2.bitwise_or(mask_w,mask_y)
        mask_yw_image = cv2.bitwise_and(gray,mask_yw);
        cv2.imshow('filterLoad',mask_yw_image)
        return mask_yw_image

    
    @staticmethod
    def parseLine(img):
        lanes = cv2.HoughLines(img, 1, np.pi/180,80,10,150)
        if lanes is None:
            return []
        lines = []
        for lane in lanes:
            rho = lane[0][0]
            theta = lane[0][1]
            if(abs((theta)<50)):
                continue
            if(abs(theta)>8):
                continue

            a= np.cos(theta)
            b= np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 =int( x0 + 1000*(-b))
            y1 =int( y0 + 1000*(a))
            x2 =int( x0 - 1000*(-b))
            y2 =int( y0 - 1000*(a))
            lines.append((x1,y1,x2,y2))


        return lines
    threshold  = 40
    minLineLength = 60
    maxLineGap =151
    rho = 1
    @staticmethod
    def printCurrentHoughParam():
        print('threshold:{}\nminLineLength:{}\nmaxlineGap:{}\nrho:{}\n'.
              format(LaneInfo.threshold,LaneInfo.minLineLength,LaneInfo.maxLineGap,LaneInfo.rho))
    @staticmethod
    def parseLineP(img):
        lanes = cv2.HoughLinesP(img,LaneInfo.rho, np.pi/180,threshold=LaneInfo.threshold,minLineLength=LaneInfo.minLineLength,maxLineGap= LaneInfo.maxLineGap)
        if lanes is None:
            return []
        lines = []
        for lane in lanes:
           
            lines.append(lane)


        return lines

    #def __init__(self, *args, **kwargs):
   #     return super(LaneInfo,self).__init__(*args, **kwargs)

class ImageProcesser:
    def processImage( lane, obstacle, car, human, trafficLight, trafficSignal):
     pass


    
class ObstacleInfo:
    isblocking = True;
    distance=0
    relativeSpeed=0
    absoluteSpeed=0
    pass
    
class Car(ObstacleInfo):
    pass

class Human(ObstacleInfo):
    pass

class TrafficLight:
    pass

class TrafficSignal:
    pass


class DriveState:
    speed=0
    obstacle = []
    lane = LaneInfo()

class Marker:
    pass

class Controler:
    def alart():
        pass

def updateState():
    Controler.alart()

  
                     
class ImageView:


    def __init__(self, imgpath=None,img=None):
        if imgpath is not None:
            self.img = cv2.imread(imgpath)
        elif img is not None:
            self.img = img
        self.r = (37, 235, 524, 125)
        

        pass

    #adjust here for threshing tangent
    def parseAndRenderLine(self, edge_img, target_img):
        lines =LaneInfo.parseLineP(edge_img)
        for point in lines:
            for x1,y1,x2,y2 in point:
                a = (y2-y1)*1.0/(x2-x1)
                a = abs(a)
                if(a<0.6):
                    continue
                elif(a>10):
                    continue

                cv2.line(target_img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('line_recog',target_img)
        return target_img

    def parseAndRenderLineRectROIMark(self, roi_edge,target_img):
        lines =LaneInfo.parseLineP(roi_edge)
        for point in lines:
            for x1,y1,x2,y2 in point:
                x1 = self.r[0]+x1
                y1 = self.r[1]+y1
                x2 = self.r[0]+x2
                y2 = self.r[1]+y2
                cv2.line(target_img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('line_recog',target_img)
        return target_img


    #roi 처리를 이함수안에서 처리함
    def renderLineROIProc(self, edge_img,target_img):
        roi = self.cropRectROI(edge_img)
        roi_edge = VideoView.detectEdge(roi);
        lines =LaneInfo.parseLineP(roi_edge)
        for point in lines:
            for x1,y1,x2,y2 in point:
                x1 = self.r[0]+x1
                y1 = self.r[1]+y1
                x2 = self.r[0]+x2
                y2 = self.r[1]+y2
                cv2.line(target_img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow('line_recog',target_img)
        return target_img
 
   
    def processImg(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        contrast = VideoView.histoEqualization(gray)
        ImageView.showFrame('contrast',contrast)
        lane = LaneInfo.filterLoad(img,contrast)
        lane = VideoView.gaussianBlur(lane)
        line_image = VideoView.detectEdge(lane) 
        result = self.renderLineROIProc(line_image,img.copy())
        #result = ImageView.weighted_img(img,result,0.8,1,0)
        #cv2.imshow('weighted',result)

    def processImgROI(self,img):
        roi = self.cropFreeROI(img)
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        gray = ImageView.sharpening(gray)

        contrast = VideoView.adaptiveHisto(gray) #histogram equalization
        ImageView.showFrame('contrast',contrast)
        contrast = self.cropFreeROI(contrast)
        lane = LaneInfo.filterLoad(roi,gray,180)
        lane = VideoView.gaussianBlur(lane)
        lane = self.cropFreeROI(lane)
        line_image = VideoView.detectEdge(lane) 

        result = self.parseAndRenderLine(line_image,img.copy())

    def processCommand(self,key):
        if key== ord('q'):
            pass
        elif key == ord('w'):
            pass
        elif key == ord('r'):
            self.selectArea(self.img)
        elif key ==ord('t'):
            VideoView.showHisto(self.img)
        elif key == ord('f'):
            cur_frm = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            rel_pos = self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,cur_frm + 1000)
            print(rel_pos)
        elif key == ord('s'):
            #cv2.imwrite(r'F:\Downloads\saved.jpg',img);
            pass
        elif key == ord('y'):
            LaneInfo.rho = LaneInfo.rho+1
        elif key == ord('u'):
            LaneInfo.rho = LaneInfo.rho-1
        elif key == ord('i'):
            LaneInfo.threshold = LaneInfo.threshold+1
        elif key == ord('o'):
            LaneInfo.threshold = LaneInfo.threshold-1
        elif key == ord('p'):
            LaneInfo.maxLineGap = LaneInfo.maxLineGap+1
        elif key == ord('h'):
            LaneInfo.maxLineGap = LaneInfo.maxLineGap-1
        elif key == ord('j'):
            LaneInfo.minLineLength = LaneInfo.minLineLength+1
        elif key == ord('k'):
            LaneInfo.minLineLength = LaneInfo.minLineLength-1
        elif key == ord('l'):
            LaneInfo.rho = LaneInfo.rho+1
        elif key == ord(';'):
            LaneInfo.rho = LaneInfo.rho-1
        elif key==ord('n'):
            try:
                self.img_idx +=1
            except:
                self.img_idx = 2

            impath = "{:08d}.jpg".format(self.img_idx)
            self.img = cv2.imread(impath)


        LaneInfo.printCurrentHoughParam()
    @staticmethod
    def processMouseEvent(event,x,y,flags,param):
        if(flags==1):
            print('event:{}\nx:{}\ny:{}\nflags:{}\nparam:{}\n'.format(event,x,y,flags,param))
        pass
  
    def run(self,delay=0):
        while True:
            key = cv2.waitKey(delay)
            self.processCommand(key)
            cv2.imshow('original',self.img)
            cv2.setMouseCallback('original',ImageView.processMouseEvent)
            self.processImgROI(self.img)
    
    def show(self,time):
          key = cv2.waitKey(time)
          self.processCommand(key)
          cv2.imshow('original',self.img)
          cv2.setMouseCallback('original',ImageView.processMouseEvent)
          self.processImgROI(self.img)


    def cropRectROI(img,r):
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        cv2.destroyWindow('ROI selector')
        return imCrop;
    def cropRectROI(self, img):
        imCrop = img[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])]
        cv2.destroyWindow('ROI selector')
        return imCrop;
    #polygon crop - mask & bitwise and 활용
    def cropFreeROI(self,r):
        size = self.img.shape
        mask = np.zeros(size ,dtype = np.uint8)
        edge1 = [(36,350),(214,234),(273,235),(613,352)]
        edge2 = [(3,357),(2,306),(209,240),(300,240),(635,340),(635,357)]
        edge3 = [(4,356),(1,282),(246,198),(297,199),(639,317),(639,357)]
        edge4 = [(4,479),(1,282),(246,198),(297,199),(639,317),(639,479)]
        
        roi_corners = np.array([edge4],dtype = np.int32)

        height = size[0]
        width = size[1]
        roi_corners = [
		(0, height),
		(width*1 / 4, height*1 / 3),
		(width*3 / 4, height*1 / 3),
		(width, height),
	    ]
        roi_corners = np.array([roi_corners],dtype=np.int32)

        channel_count = self.img.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask,roi_corners,ignore_mask_color)
        cv2.imshow('mask',mask)
        masked = cv2.bitwise_and(self.img,mask)
        return masked
    
    def selectArea(self,img):
        self.r= cv2.selectROI(img)
        imCrop = img[int(self.r[1]):int(self.r[1]+self.r[3]), int(self.r[0]):int(self.r[0]+self.r[2])]
        print("roi: " + str(self.r))
        return imCrop;

    def readFrame(self):
        if not self.cap.isOpened():
            return None
        return self.cap.read()

    @staticmethod
    def showFrame(name, img):
        cv2.imshow(name,img)
    @staticmethod
    def erosion(img):
        kernel = np.ones((10,10),np.uint8)
        result = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow("erosion",result)
        return result
    @staticmethod
    def dilation(img):
        kernel = np.ones((10,10),np.uint8)
        result = cv2.dilate(img,kernel,iterations = 1)
        cv2.imshow("dilate",result)
        return result
    @staticmethod
    def detectEdge(img):
        edge = cv2.Canny(img,50,150);
        cv2.imshow('detect-edge',edge)
        return edge
    @staticmethod
    def sharpening(img):
        kernel = np.zeros((9,9),np.float32);
        kernel[4,4] = 2.0
        boxFilter = np.ones((9,9),np.float32)/81.0
        kernel = kernel - boxFilter
        custom = cv2.filter2D(img,-1,kernel)
        cv2.imshow('sharp',custom)
        return custom
    @staticmethod
    def gaussianBlur(img):
        blur = cv2.GaussianBlur(img,(3,3),0)
        cv2.imshow('g-blur',blur)
        return blur

    @staticmethod
    def showHisto(img):
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normal = cdf*hist.max()/cdf.max()
        plt.plot(cdf_normal, color = 'b')
        plt.hist(img.flatten(),256,[0,256],color='r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'),loc = 'upper left')
        plt.show()
    @staticmethod
    def clahe(img):
        dst = cv2.createCLAHE()
    @staticmethod
    def histoEqualization(img):
         hist,bins = np.histogram(img.flatten(),256,[0,256])
         cdf = hist.cumsum()
         cdf_normal = cdf*hist.max()/cdf.max()
         cdf_m = np.ma.masked_equal(cdf,0);
         cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
         cdf = np.ma.filled(cdf_m,0).astype('uint8')
         cv2.imshow('histoEq',cdf[img])
         return cdf[img]

    @staticmethod
    def adaptiveHisto(img):
        clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
        return clahe.apply(img)

    speed = 1;




   
class VideoView(ImageView):

    def __init__(self,path):
        self.videopath = path
        self.openVideo()
    
    def openVideo(self):
        self.cap = cv2.VideoCapture(self.videopath)
        

    def processCommand(self,key):
            if key== ord('q'):
                VideoView.speed = VideoView.speed+10
                print(VideoView.speed)
            elif key == ord('w'):
                VideoView.speed = VideoView.speed-10
                print(VideoView.speed)
            elif key == ord('r'):
                self.selectArea(self.original)
                
            elif key ==ord('h'):
                VideoView.showHisto(original)
            elif key == ord('f'):
                cur_frm = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                rel_pos = self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,cur_frm + 1000)
                print(rel_pos)
            elif key == ord('s'):
                cv2.imwrite(r'F:\Downloads\saved.jpg',self.original);

    

    def playVideo(self):
        self.r = (93, 176, 409, 184)
        
        while True:
            key = cv2.waitKey(VideoView.speed)
            self.processCommand(key)


            result = self.readFrame()
            if result is None:
                break
            self.original = result[1]
            original  = result[1];
            gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)


            try:
                pass
            except Exception as e:
                pass

            ivew = ImageView(img= original)
            ivew.r = self.r
            ivew.processImgROI(ivew.img)
            #VideoView.showFrame('original',original)
            #VideoView.showFrame('h-e',VideoView.histoEqualization(gray))
            #VideoView.showFrame('a-h',VideoView.adaptiveHisto(gray))
            #adaptive =  VideoView.adaptiveHisto(gray);
            #VideoView.showFrame('a-h-edge',VideoView.detectEdge(adaptive))
            #VideoView.showFrame('edge',VideoView.detectEdge(gray))
            #contrast = VideoView.adaptiveHisto(gray);
            #contrast = ImageView.histoEqualization(gray)
            

            #blur = VideoView.gaussianBlur(sharp)
            #VideoView.showFrame('blur',blur);
            #original_clone = original.copy();
            #try:
            #    if self.r is not None:
            #        roi = VideoView.cropRectROI(line_image,self.r)
            #        VideoView.showFrame('roi',roi)
            #        lines =laneInfo.parseLineP(roi)
            #        for point in lines:
            #            for x1,y1,x2,y2 in point:
            #                x1 = self.r[0]+x1
            #                y1 = self.r[1]+y1
            #                x2 = self.r[0]+x2
            #                y2 = self.r[1]+y2
            #                cv2.line(original_clone,(x1,y1),(x2,y2),(0,0,255),2)
            #        #weighted = ImageView.weighted_img(original,original_clone)
            #        VideoView.showFrame('original',original_clone)
            
            #except Exception as e:
            #     for point in laneInfo.parseLineP(line_image):
            #         cv2.line(original,(point[0],point[1]),(point[2],point[3]),(0,0,255),2)
            #         VideoView.showFrame('original',original_clone)
  