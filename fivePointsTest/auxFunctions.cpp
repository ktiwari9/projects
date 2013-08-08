#include "auxFunctions.h"

using namespace cv;
using namespace std;

/******************************************************************
        DISPLAY FUNCTIONS
*******************************************************************/
void print(Mat X)
{
    for(int i=0; i<X.rows; ++i)
    {
        for(int j=0; j<X.cols; ++j)
            cout << X.at<double>(i,j) << " ";
        cout << endl;
    }
}

void print(libmv::Mat X)
{
    for(int i=0; i<X.rows(); ++i)
    {
        for(int j=0; j<X.cols(); ++j)
            cout << X(i,j) << " ";
        cout << endl;
    }
}

void print(libmv::Mat3 X)
{
    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
            cout << X(i,j) << " ";
        cout << endl;
    }
}

void print(libmv::Mat2X X)
{
    for(int i=0; i<2; ++i)
    {
        for(int j=0; j<X.cols(); ++j)
            cout << X(i,j) << " ";
        cout << endl;
    }
}

