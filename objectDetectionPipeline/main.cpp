#include "Detector.hpp"

// the learning pipeline is made of
//              feature detector
//              feature descriptor
//              feature processor (e.g. bow)
//              learning stage (e.g. svm)
//              learning processor (e.g. tesselation)

// the detection pipeline is made of
//              feature detector
//              feature descriptor
//              feature processor (e.g. bow)
//              classification stage
int main(int argc, char** argv)
{
    DetectorBlackBox* d = new DetectorBlackBox(SIFT_ENUM);
    DetectorBlackBox* d = new DetectorBlackBox(SIFT_ENUM);

    if ( d->readImage("/home/gmanfred/Pictures/engineer.jpg") == 0 )
    {
        d->extractKeyPoints();
        d->showKeyPoints();
    }

    waitKey(0);

    return 0;
}
