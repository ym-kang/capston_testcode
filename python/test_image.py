#get image for test

import cv2

base = '../test_data/dataset/car1/'
ground = 'groundtruth.txt'
index = 0

def readImgTxt():
    img_name =[]
    f = open(base + 'images.txt','r')
    img_name = f.readlines()
    
    for i in range(len(img_name)):
        img_name[i] = img_name[i].replace("\n","")
        img_name[i] = img_name[i].replace("\r","")
    return img_name


imgs = readImgTxt()




def toBoxList(bbox_line):
    vals = bbox_line.split(",")
    bboxs = []
    i=0
    while(i<len(vals)):
        bboxs.append((
                int(float(vals[i])),
                int(float(vals[i+1])),
                int(float(vals[i+2])),
                int(float(vals[i+3]))
                ))
        i = i+4
    return bboxs

def getboxes(index):
    f = open(base + ground)
    for i,line in enumerate(f):
        if i==index:
            bboxs = line.replace("\n","")
            bbox = toBoxList(bboxs)
            break

    return bbox

def drawbbox(img, bboxs):
    for i,bbox in enumerate(bboxs):
        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)

def showImgFile(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    cv2.imshow('Disp',img)
        
    return img
