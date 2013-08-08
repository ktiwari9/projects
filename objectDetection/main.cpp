#include <iostream>
#include <vector>
#include <cstdio>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "FindCameraMatrices.h"
#include "Triangulation.h"

#define EPSILON 0.0001

#define DET "SIFT"
#define DESC "SURF"
#define MATCH "FlannBased"
//#define MATCH "BruteForce-Hamming"
//#define MATCH "BruteForce"

using namespace std;
using namespace cv;

void print(Mat X)
{
    for(int i=0; i<X.rows; ++i)
    {
        for(int j=0; j<X.cols; ++j)
            cout << X.at<double>(i,j) << " ";
        cout << endl;
    }
}

void print(Matx34d X)
{
    for(int i=0; i<X.rows; ++i)
    {
        for(int j=0; j<X.cols; ++j)
            cout << X(i,j) << " ";
        cout << endl;
    }
}

int main(int argc, char** argv)
{
    // fake input for testing
    argc = 3;
    argv[1] = "/home/gmanfred/Desktop/leftRect0000.jpg";
    argv[2] = "/home/gmanfred/Desktop/rightRect0000.jpg";

    //argv[1] = "/home/gmanfred/Desktop/rightRect0000.jpg";
    //argv[2] = "/home/gmanfred/Desktop/leftRect0000.jpg";

    //argv[1] = "/home/gmanfred/right1.jpg";
    //argv[2] = "/home/gmanfred/right2.jpg";
    //argv[1] = "/home/gmanfred/Desktop/leftRect0000.jpg";
    //argv[1] = "/home/gmanfred/Desktop/rightRect0000.jpg";
    //argv[2] = "/home/gmanfred/Desktop/rightRect0000P60.jpg";

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

    cout << "Detect keypoints -start-" << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create(DET);
    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;

    detector->detect(img1, kpts1);
    detector->detect(img2, kpts2);
    if(kpts1.empty() || kpts2.empty()){
        cout << "No keypoints found" << endl;
        return -3;
    }
    cout << "Detect keypoints -end-" << endl;


    cout << "Describe keypoints -start-" << endl;
    Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(DESC);
    Mat descs1;
    Mat descs2;

    descriptor->compute(img1, kpts1, descs1);
    descriptor->compute(img2, kpts2, descs2);
    if(descs1.empty() || descs2.empty()){
        cout << "Descriptor vector null" << endl;
        return -4;
    }
    cout << "Describe keypoints -end-" << endl;


    cout << "Match descriptors -start-" << endl;
    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(MATCH);

    matcher->match(descs1, descs2, matches);

    double max_dist = 0; double min_dist = 100;
     //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descs1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Keep only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descs1.rows; i++ )
    {
        if( matches[i].distance < 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    int nb_matches = good_matches.size();
    cout << "Nb matches :" << nb_matches << endl;
    if(nb_matches == 0){
        cout << "No matches found" << endl;
        return -5;
    }

    cout << "Match descriptors -end-" << endl;


    cout << "Save good matches -start-" << endl;
    vector<Point2f> pts1(nb_matches);
    vector<Point2f> pts2(nb_matches);
    vector<int> queryIdxs( nb_matches ), trainIdxs( nb_matches );

    for( size_t i = 0; i < nb_matches; ++i )
    {
        queryIdxs[i] = good_matches[i].queryIdx;
        trainIdxs[i] = good_matches[i].trainIdx;
    }
    KeyPoint::convert(kpts1, pts1, trainIdxs);
    KeyPoint::convert(kpts2, pts2, queryIdxs);

    FILE* stream1 = fopen("keypoints1", "w+");
    for(int i=0; i<pts1.size(); ++i)
        fprintf(stream1, "%lf %lf %lf\n", pts1[i].x, pts1[i].y, 1.0);

    fclose(stream1);

    FILE* stream2 = fopen("keypoints2", "w+");
    for(int i=0; i<pts1.size(); ++i)
        fprintf(stream2, "%lf %lf %lf\n", pts2[i].x, pts2[i].y, 1.0);
    fclose(stream2);
    cout << "Save good matches -end-" << endl;
    /*
    vector<Point2d> pts1_good,
					pts2_good;

    Matx34d P1,P2;
    Mat cam_matrix1, cam_matrix2;
    Mat K1, K2;
    Mat_<double> K1inv, K2inv;

    P1= cv::Matx34d(1,0,0,0,
					0,1,0,0,
					0,0,1,0);
	P2 = cv::Matx34d(1,0,0,50,
					 0,1,0,0,
					 0,0,1,0);

	cv::FileStorage fs1,fs2;
	fs1.open("/home/gmanfred/Desktop/l_camera_calibration.yml",cv::FileStorage::READ);
	fs2.open("/home/gmanfred/Desktop/r_camera_calibration.yml",cv::FileStorage::READ);
	fs1["camera_matrix"]>>cam_matrix1;
	fs2["camera_matrix"]>>cam_matrix2;
	K1 = cam_matrix1;
	K2 = cam_matrix2;
    invert(K2, K2inv); //get inverse of camera matrix
    invert(K1, K1inv); //get inverse of camera matrix
    print(K1);

    FindCameraMatrices(K1, K2, K1inv, K2inv, pts1, pts2, pts1_good, pts2_good, P1, P2);
    cout << "Find camera matrices -end-" << endl;


    cout << "Backprojecting 3D points -start-" << endl;

    vector<Point2d> reproPts;
    for(int i=0; i<point_cloud.size(); ++i)
    {
        Mat pt4d = (Mat_<double>(4,1) << point_cloud[i].x, point_cloud[i].y, point_cloud[i].z, 1);
        print(pt4d);
        Mat pt = Mat(P2)*pt4d;
        reproPts.push_back(Point2d(pt.at<double>(0,0), pt.at<double>(1,0)));
    }

    // draw reprojected points along with keypoints position
	Mat tmp(640,480,CV_8UC3); cvtColor(img1, tmp, CV_BGR2HSV);
	for (unsigned int i=0; i<reproPts.size(); ++i) {
		circle(tmp, reproPts[i], 1, Scalar(200,255,255), CV_FILLED);
        //circle(tmp, pts2[i], 1, Scalar(0,255,255), CV_FILLED);
	}
	cvtColor(tmp, tmp, CV_HSV2BGR);
	imshow("Back projection", tmp);
	waitKey(0);
	destroyWindow("Back projection");

    cout << "Backprojecting 3D points -end-" << endl;


    cout << "Bundle adjustement -start-" << endl;
    cout << "Bundle adjustement -end-" << endl;

    return 0;
    */
}
