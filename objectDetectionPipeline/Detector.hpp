#ifndef DEF_DETECTORBLACKBOX
#define DEF_DETECTORBLACKBOX

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

enum detectorType {SIFT_ENUM,SURF_ENUM,STAR_ENUM,MSER_ENUM,FAST_ENUM};

class DetectorBlackBox
{
    public:

        DetectorBlackBox(detectorType type);

        int readImage(char* img);
        int extractKeyPoints();
        void showKeyPoints();
        vector<KeyPoint> getKeyPoints();

    protected:

        FeatureDetector* _myDetector;
        vector<KeyPoint> _myKeypoints;
        Mat _myImg;
};

#endif // DEF_DETECTORBLACKBOX

