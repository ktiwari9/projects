#include <iostream>
#include <cstdio>

#include "libmv/multiview/five_point.h"
#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "auxFunctions.h"
#include "Matcher.h"
#include "FivePointsWrapper.h"
#include "TriangulationFunctions.h"

#define DET "SURF"
#define DESC "SURF"
#define MATCH "FlannBased"
//#define DET "FAST"
//#define DESC "BRIEF"
//#define MATCH "BruteForce-Hamming"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    argc = 3;
    //argv[1] = "/home/gmanfred/right1.jpg";
    //argv[2] = "/home/gmanfred/right2.jpg";
    argv[1] = "/home/gmanfred/Desktop/leftRect0000.jpg";
    argv[2] = "/home/gmanfred/Desktop/rightRect0000.jpg";

    if(argc != 3){
        cout << "Usage: objectDetection path_to_image1 path_to_image2" << endl;
        return -1;
    }

    cout << "Load images -start-" << endl;
    Mat img1= imread(argv[1]);
    Mat img2= imread(argv[2]);
    if(img1.empty() || img2.empty()){
        cout << "Images not found" << endl;
        return -2;
    }
    cout << "Load images -end-" << endl;


    cout << "Matching features -start-" << endl;
    Matcher matcher(DET, DESC, MATCH);
    matcher.match(img1, img2);
    cout << "Matching features -end-" << endl;


    cout << "Calibrate points -start-" << endl;
    // Camera matrix without the last line as our points are not in homogeneous space
    Mat K1(3, 3, CV_64FC1);
    cv::FileStorage r_fs;
    r_fs.open("/home/gmanfred/Desktop/r_camera_calibration.yml",cv::FileStorage::READ);
    r_fs["camera_matrix"]>>K1;
    matcher.calibratePoints(K1);
    cout << "Calibrate points -end-" << endl;


    cout << "Five points -start-" << endl;
    Mat_<double> Ra(3,3);
    Mat_<double> Rb(3,3);
    Mat_<double> t(3,1);
    FivePoints fpAlgo;

    fpAlgo.compute(matcher._calib_hpts1, matcher._calib_hpts2, Ra, Rb, t);
    //cout << fpAlgo.getMinError() << endl;
    cout << "Five points -end-" << endl;


    cout << "Get pose matrix -start-" << endl;
    Matx34d P;
    Point3d chpt1 = Point3d(matcher._calib_hpts1.at<double>(0,0), matcher._calib_hpts1.at<double>(1,0), matcher._calib_hpts1.at<double>(2,0));
    Point3d chpt2 = Point3d(matcher._calib_hpts2.at<double>(0,0), matcher._calib_hpts2.at<double>(1,0), matcher._calib_hpts2.at<double>(2,0));
    solveChirality(chpt1, chpt2, Ra, Rb, t, P);
    cout << "Get pose matrix -end-" << endl;

    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            printf("%lf\n", P(i,j));
    return 0;
}
