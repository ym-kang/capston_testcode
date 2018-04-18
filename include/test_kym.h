#define OPENCV
#include "darknet.h"
#include "math.h"
#include "python2.7/Python.h" 

#ifdef __cplusplus
extern "C"
{
#endif


image ipl_arr_to_image(int w,int h,int c,float* arr);


image getImg();

void initializeCapture(char* fileName);

void initializeCam(int index);

IplImage* getIpl();


void destroy_cap();


void show(image img);

#ifdef __cplusplus
}
#endif
int cvRound(double value);// {return(ceil(value));}

