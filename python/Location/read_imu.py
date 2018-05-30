
port = "/dev/ttyACM0"
baud_rate = 115200
imu_mode = "AMGQUA"

import serial

def readImu():
    ser = serial.Serial(port)
    print(ser.name)
    ser.write("@"+imu_mode+"\r\n")
    
    while(True):
        print(ser.readline())

readImu()