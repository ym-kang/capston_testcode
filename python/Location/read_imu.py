
port = "/dev/ttyACM0"

baud_rate = 115200
imu_mode = "AMGQUA"
cmd = "AMGQUA"
import serial
ser = serial.Serial(port,baud_rate)

def readImu():
    while(True):
        print(ser.readline())

readImu()