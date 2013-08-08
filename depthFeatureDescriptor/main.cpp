#include <iostream>
#include <ctime>
#include "DepthFeatureDescriptor.h"

#define NUM_PIXELS_ANALYZED 2000
#define NUM_FEATURES_CANDIDATE
#define OUT_OF_BOUNDS 10000

class MyPoint
{
public:
    myPoint(float x, float y)
    {
        this->x = x;
        this->y = y;
    }

    operator+(myPoint p)
    {
        this->x += p.x;
        this->y += p.y;
    }

    operator/(float s)
    {
        this->x /= s;
        this->y /= s;
    }

    float x;
    float y;
}

float computeFeatureXY(const Mat d, MyPoint x, MyPoint u, MyPoint v)
{
    float f = 0;
    float fU = 0;
    float fV = 0;

    CvPoint tmpU = x + ( u/float(d(x.x, x.y)) );
    CvPoint tmpV = x + ( v/float(d(x.x, x.y)) );

    // the selected u takes a point out of bounds.
    if( tmpU.x > d.cols || tmpU.y > d.rows || tmpU.x < 0 || tmpU.y < 0 )
        fU = OUT_OF_BOUNDS;
    else
        fU = d(tmpU.x, tmpU.y);

    // the selected v takes a point out of bounds.
    if ( tmpV.x > d.cols || tmpV.y > d.rows || tmpV.x < 0 || tmpV.y < 0 )
        fV = OUT_OF_BOUNDS;
    else
        fV = d(tmpV.x, tmpV.y);

    f = fU - fV;

return f;
}

float* computeFeatures(const Mat d)
{
    float* features = (float*)malloc(NUM_PIXELS_ANALYZED*sizeof(float));
    MyPoint x(0,0);
    MyPoint u(0,0);
    MyPoint v(0,0);

    int thetaStep = 1;

    srand ( time(NULL) );

    for(int count=0; count< NUM_PIXELS_ANALYZED; ++x)
    {
        // Randomly select a pixel
        x.x = rand() % d.cols;
        x.y = rand() % d.rows;
        for(int countTheta=0; countTheta< nbU; countTheta+=thetaStep)
        {
            // Randomly select two feature vector u v.
            u.x = rand() % d.cols;
            u.y = rand() % d.rows;
            v.x = rand() % d.cols;
            v.y = rand() % d.rows;
            features[count] = computeFeatureXY(d, x, u, v);
        }
    }
}

int main()
{


    //computeFeatureXY(const Mat I, int x, int y, int u, int v);

    return 0;
}
