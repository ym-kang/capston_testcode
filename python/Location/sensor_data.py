#-*- coding: utf-8 -*-

import math

#imu 데이터를 보관할 클래스
class IMUData(object):
    #오일러 각 https://ko.wikipedia.org/wiki/%EC%98%A4%EC%9D%BC%EB%9F%AC_%EA%B0%81

    #imu 모듈 데이터기준으로 생성
    def __init__(self,roll,pitch,yaw):
        self.yaw = yaw 
        self.pitch = pitch
        self.roll = roll
        
        self.r_yaw =math.radians(yaw)  #yaw radian
        self.r_pitch = math.radians(pitch) #picth radian
        self.r_roll = math.radians(roll)#roll radian
        
#gps 데이터를 보관할 클래스
class GPSData(object):
    
    #gps 모듈 데이터기준으로 생성
    def __init__(self,latitude,longitude,altitude):
        self.longitude = longitude #경도
        self.latitude = latitude #위도
        self.altitude = altitude #고도
        #http://lovestudycom.tistory.com/entry/%EC%9C%84%EB%8F%84-%EA%B2%BD%EB%8F%84-%EA%B3%84%EC%82%B0%EB%B2%95
        #https://m.blog.naver.com/PostView.nhn?blogId=bkpark777&logNo=80142944295&proxyReferer=https%3A%2F%2Fwww.google.com%2F
        #적도에서 위도가 1도 변할때 길이는 110.569km이라고 한다.
        self.length_per_latitude = 110.569*1000 
        #경도가 1도 변할때 길이는 111.322km*cos(위도)로 근사한다.
        self.length_per_longitude = 111.322*1000*math.cos(math.radians((self.latitude)))
    
    #매개변수 - 대상 물체의 xy평면 거리 및 방위각(degree)
    def calculateLoc(self, distance, angle):
        angle = math.radians(angle) #degree에서 radian 단위로 변환
        #물체의 위도 계산
        lat = self.latitude + math.cos(angle) * distance / self.length_per_latitude
        #물체의 경도 계산
        lon = self.longitude + math.sin(angle) * distance / self.length_per_longitude
        #고도 계산 생략
        

        #물체의 실제 좌표 반환
        return (lat,lon)


#카메라 좌표 정보를 보관할 클래스
class CameraMath(object):
    #ocam spec: http://withrobot.com/camera/ocams-1cgn-u/
    #static variables
    field_of_view_horizental = 92.8  #FOV - ocam 공식 홈페이지 스펙 참조
    filed_of_view_vertical = 50.0 
    horizental_resoultion = 1280 #해상도
    vertical_resolution = 960
    center_x = horizental_resoultion/2 #중심좌표
    center_y = vertical_resolution/2
    angle_per_x_pixel = field_of_view_horizental/horizental_resoultion #픽셀당 각 변화량
    angle_per_y_pixel = filed_of_view_vertical/vertical_resolution

    #생성시 x,y 픽셀 정보 거리, 이름을 매개변수로준다.
    def __init__(self,x_pixel,y_pixel,distance,name):
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.distance = distance
        self.name = name

#calculate angle in pixel
#논문: 단일 카메라와 GPS를 이용한 영상 내객체 위치 좌표 추정 기법
    def calculatePixelAngle(self): 
        x_angle = CameraMath.field_of_view_horizental/CameraMath.horizental_resoultion*(self.x_pixel-CameraMath.center_x)
        y_angle = CameraMath.filed_of_view_vertical/CameraMath.vertical_resolution*(self.y_pixel-CameraMath.center_y)
        return (x_angle,y_angle)

def calculatePixelAngle(x_pixel,y_pixel,distance,name):
    obj = CameraMath(x_pixel,y_pixel,distance,name)
    return obj.calculatePixelAngle()

#회전변환 행렬 
#https://ko.wikipedia.org/wiki/%ED%9A%8C%EC%A0%84%EB%B3%80%ED%99%98%ED%96%89%EB%A0%AC
#https://m.blog.naver.com/PostView.nhn?blogId=kimjw1218&logNo=70178629876&proxyReferer=https%3A%2F%2Fwww.google.com%2F
import numpy
#위치 계산! - imu data, gps data, (거리 정보, 물체의 x픽셀좌표, y픽셀 좌표 필요 - 카메라 클래스에 포함)
def calculateLocation(imu_data,gps_data, camera):
    #먼저 화면내 픽셀의 각도를 계산한다
    pixel_angle_x, pixel_angle_y = camera.calculatePixelAngle()
    print("pixel angle: ",pixel_angle_x,pixel_angle_y)
    
    angle_matrix = [[pixel_angle_x],[pixel_angle_y]] 
    #roll 값에 따라 x,y 각을 회전변환한다 (구면 좌표계 참조).
    roll_matrix = [[math.cos(imu_data.r_roll),-math.sin(imu_data.r_roll)],
    [math.sin(imu_data.r_roll),math.cos(imu_data.r_roll)]]
    result = numpy.matmul(roll_matrix,angle_matrix)
    #yaw 값에 따라 x 각을 회전시킨다.
    result = result + [[imu_data.yaw],[0]]
    #pitch 값에 따라 y 각을 회전시킨다.
    result = result + [[0],[imu_data. pitch]] 
    #방위각 계산 부분
    #pi_angle = imu_data.yaw + pixel_angle_x*math.cos(imu_data.roll) + pixel_angle_y*math.sin(imu_data.roll) # 방위각 구할것
    pi_angle = result[0][0]  
    theta_angle = result[1][0]
    #xy 평면상의 거리로 변환해준다.
    xy_plane_distance = math.cos(math.radians(theta_angle))*camera.distance
    print("xy_plane_distance: ",xy_plane_distance)
    #변환된 픽셀의 각도
    print("rotated pixel angle: ",pi_angle,theta_angle)
    
    lat,lng = gps_data.calculateLoc(xy_plane_distance,pi_angle)
    print("lat,lng: ",lat,lng)
    return (lat,lng) #픽셀의 위도 및 경도 반환
   

def test1():
    imu = IMUData(0,0,0)
    gps = GPSData(35.886906,128.60928,0)
    dist = 100
    x = 640
    y = 480
    camera = CameraMath(x,y,dist,"name")
    calculateLocation(imu,gps,camera)

#360도 회전하며 100m 전방 위치 계산
def test10():
    imu = IMUData(0,0,0)
    gps = GPSData(35.886906,128.60928,0) #기준점
    dist = 100
    x = 640
    y = 480
    camera = CameraMath(x,y,dist,"name2")
    results = []
    for i in range(10):
        imu = IMUData(0,0,36*i) #회전시 imu의 yaw값이 바뀌게 됨
        results.append(calculateLocation(imu,gps,camera))

    for r in results:
        format = "{{lat:{}, lng: {}}},"
        print(format.format(r[0],r[1]))
    return results


cameraDatas = [] #카메라 객체정보 리스트
imu = IMUData(0,0,0) #imu 센서 data update해줄것.
gps = GPSData(35.886906,128.60928,0) #gps 센서 data update해줄것.


#json형식의 리스트로 반환
def calculateAll():
    global imu,gps,cameraDatas
    results = []
    #탐지한 물체들의 좌표를 계산한다.
    for camData in cameraDatas:
        results.append([calculateLocation(imu,gps,camData),camData.name])

    jsondatas = []
    #json형식으로 변환
    for result in results:
        jsondata = {'lat':result[0][0],'lng':result[0][1],'name':result[1]}  
        jsondatas.append(jsondata)

    jsondatas.append({'lat':gps.latitude,'lng':gps.longitude,'name':'jetsonTX2'})
    return jsondatas
