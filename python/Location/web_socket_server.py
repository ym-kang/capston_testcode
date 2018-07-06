#-*- coding: utf-8 -*-

#웹에서 요청하는 위치정보 전송 코드
#https://coderwall.com/p/gvknya/python-websocket-server

from tornado import websocket
import tornado.ioloop

import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web

import socket
'''
This is a simple Websocket Echo server that uses the Tornado websocket handler.
Please run `pip install tornado` with python of version 2.7.9 or greater to install tornado.
This program will echo back the reverse of whatever it recieves.
Messages are output to the terminal for debuggin purposes. 
''' 
names = ['flare_signal','warning_tripod','dog','cat','deer','pedestrian','car','fire_truck','ambulance']
import json
import random
import sensor_data
class ObjectInfo:#좌표정보를 json으로 변환
    def __init__(self):
        pass
        #self.objList = objList
    use_random = False
    
    def toMessage(self):#none -> random data

        if(ObjectInfo.use_random): #랜덤 모드인경우 -> 랜덤 좌표 전송
            objList = []  #개별 항목은 lat, lng, name로 구성된  json형식, list
            results = sensor_data.test10()
            for result in results:
                objList.append({'lat':result[0]+ random.random()*0.0005,
                'lng':result[1]+random.random()*0.0005,
                'name': names[(int(random.random()*len(names)))]
                })
        else:
            objList = sensor_data.calculateAll()

        
        message = json.dumps(objList)
        return message
objinfo = ObjectInfo() #탐지된 객체 데이터 정보를 담은 클래스


import threading
from time import sleep
class WSHandler(tornado.websocket.WebSocketHandler):

    
    
    def sendMsg(self):
       self.write_message(objinfo.toMessage())
            

    def open(self):
        #self.sendMsgThread = threading.Thread(target=self.sendMsg,args=(self,))
        #self.sendMsgThread.start()
       # self.pCallback = Periodical
        self.sendMsg()
        print ('new connection')

    
      
    def on_message(self, message):
        print ('message received:  %s',message)
        # Reverse Message and send it back
        #print ('sending back message: %s', message[::-1])
        print('send data')
        self.sendMsg()
        #self.write_message(message[::-1])
 
    def on_close(self):
        #self.sendMsgThread._stop()
        print ('connection closed')
 
    def check_origin(self, origin):
        return True
 
application = tornado.web.Application([
    (r'/ws', WSHandler),
])
 
 #메인 함수
def ServerThreadMain():
    
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(9000)
    myIP = socket.gethostbyname(socket.gethostname())
    print ('*** Websocket Server Started at %s***', myIP)
    tornado.ioloop.IOLoop.instance().start()
def RunServer(join = False):
    t = threading.Thread(target=ServerThreadMain)
    t.start()
    return t

if __name__ == "__main__":
    t= RunServer(True)
    t.join()
    