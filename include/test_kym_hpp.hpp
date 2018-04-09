
#include <opencv2/opencv.hpp>
#include "test_kym.h"

extern "C"{
    
image getCapMat();


cv::Mat toMat(IplImage* iplImg); 

void test(); 

void initlibkym(void);

}