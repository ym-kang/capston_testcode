#include "test_kym.h"
#include "image.h"
#include "numpy.h"
/*
void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}
*/
int cvRound(double value) {return(ceil(value));}  


//#define cvCaptureFromFile cvCreateFileCapture  

static CvCapture* cap;
static CvMat* cv_mat;

void initializeCapture(char* fileName)
{
    printf("filename:%s\n",fileName);
    cap = cvCaptureFromFile(fileName);

    if(!cap) error("Couldn't open file.\n");
//cvCapturFromFile
}
void initializeCam(int index)
{
    cap = cvCaptureFromCAM(index);
    if(!cap) error("Couldn't connect to webcam.\n");
}
static image buff [3];

image getImg()
{
    
    buff[0] = get_image_from_stream(cap);
   //if(status==0) demo_done = 1;
    return  buff[0];
}
IplImage* getIpl()
{
    return cvQueryFrame(cap);
}

void show(image image)
{
    //cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
    show_image(image,"Demo");
    int c = cvWaitKey(1);
}



void destroy_cap()
{ 
    free(cap);
}


image ipl_arr_to_image(int w,int h,int c,float* arr)
{

    image img = {0};
    img.w = w;
    img.h = h;
    img.c = c;
    img.data = malloc(sizeof(float)*w*h*c);
    int step = w * c;
    int i,j,k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                img.data[k*w*h + i*w + j] = arr[i*step + j*c + k]/255.;
            }
        }
    }


    return img;
}


image array_to_image(int w,int h,int c,float* arr)
{


    image img = {0};
    img.w = w;
    img.h = h;
    img.c = c;
    img.data = malloc(sizeof(float)*w*h*c);
    

    

    return img;
}

image videoCapture()
{

    image img = {0};


    return img;
}

/*
#from examples - detector-scipy-opencv.py
def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

*/
