from socket import *
import json
csock = socket(AF_INET, SOCK_DGRAM)
csock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
csock.bind(('', 4000))

while True:
    s, addr = csock.recvfrom(1024)
    print s
    print addr
    data = json.loads(s)
    pass

