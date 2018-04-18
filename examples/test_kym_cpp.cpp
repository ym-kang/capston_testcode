#include "test_kym_hpp.hpp"
#include <python2.7/Python.h>
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>


cv::Mat toMat(IplImage* iplImg)
{
  cv::Mat mat =  cv::cvarrToMat(iplImg);  
  
  return mat;
}


image getCapMat()
{
   
  IplImage* ipl = getIpl();
  //return toMat(ipl);
  cv::Mat mat =  cv::cvarrToMat(ipl);
  image im = {0};
  return im;
}

float* getMatArr(float* data,int w,int h,int c)
{
    IplImage* ipl = getIpl();
    cv::Mat mat = cv::cvarrToMat(ipl);
    cv::Size size = mat.size();
    int length = size.width*size.height;
    float* im = (float*)calloc(length, sizeof(float));
    for(int i=0;;i++)
    {
        im[i] = mat.data[i]/255.0;
    }
    return im;
}


void test()
{
    IplImage* ipl = getIpl();
    cv::Mat mat = cv::cvarrToMat(ipl);
    //cv::Mat mat = cv::Mat(10,10,3);
}


//test only

static PyObject* cos_func(PyObject* self, PyObject* args)
{
    double value;
    double answer;

    /*  parse the input, from python float to c double */
    if (!PyArg_ParseTuple(args, "d", &value))
        return NULL;
    /* if the above function returns -1, an appropriate Python exception will
     * have been set, and the function simply returns NULL
     */

    /* call cos from libm */
    answer = cos(value);

    /*  construct the output from cos, from c double to python float */
    return Py_BuildValue("f", answer);
}

static PyMethodDef mymethods[] = {
  {"cos_func",cos_func,METH_VARARGS,NULL},
   {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
initlibkym(void)
{
  /* Create the module and add the functions */
  Py_InitModule("libkym", mymethods);
}
/*
def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr
*/