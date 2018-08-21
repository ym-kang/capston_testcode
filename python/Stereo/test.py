import cv2
import main_stereo
main_stereo.cv_window_wait_time = 0
imgL  = cv2.imread("python/Stereo/imgs/chessboard-L6.png")
imgR =  cv2.imread("python/Stereo/imgs/chessboard-R6.png")
imgL = cv2.resize(imgL,(0,0),fx=.5,fy=.5)
imgR = cv2.resize(imgR,(0,0),fx=.5,fy=.5)

main_stereo.CalculateStereoTest(imgL,imgR)
    