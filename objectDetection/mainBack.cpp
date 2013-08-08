/**
#include <iostream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#define EPSILON 0.0001

#define DET "FAST"
#define DESC "BRIEF"
//#define MATCH "FlannBased"
#define MATCH "BruteForce-Hamming"

using namespace std;
using namespace cv;

 //From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 Mat_<double> LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                                   Matx34d P,       //camera 1 matrix
                                   Point3d u1,      //homogenous image point in 2nd camera
                                   Matx34d P1)       //camera 2 matrix
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
              );
    Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                                              -(u.y*P(2,3)  -P(1,3)),
                                              -(u1.x*P1(2,3)    -P1(0,3)),
                                              -(u1.y*P1(2,3)    -P1(1,3)));

    Mat_<double> X;
    solve(A,B,X,DECOMP_SVD);

    return X;
}


// From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997

Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                            Matx34d P,          //camera 1 matrix
                                            Point3d u1,         //homogenous image point in 2nd camera
                                            Matx34d P1)          //camera 2 matrix
{
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1);
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;

        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,
                  (u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,
                  (u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1,
                  (u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1
                  );
        Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
                          -(u.y*P(2,3)  -P(1,3))/wi,
                          -(u1.x*P1(2,3)    -P1(0,3))/wi1,
                          -(u1.y*P1(2,3)    -P1(1,3))/wi1
                          );

        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
    }
    return X;
}

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
    //argv[1] = "/home/gmanfred/Desktop/left0000.jpg";
    //argv[1] = "/home/gmanfred/Desktoxp/right0.jpg";
    //argv[2] = "/home/gmanfred/Desktop/right0000.jpg";

    argv[1] = "/home/gmanfred/Desktop/leftRect0000.jpg";
    argv[2] = "/home/gmanfred/Desktop/rightRect0000.jpg";
    //argv[1] = "/home/gmanfred/Desktop/leftRect0000P60.jpg";

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

    int nb_matches = matches.size();
    if(nb_matches == 0){
        cout << "No matches found" << endl;
        return -5;
    }

    //Mat output;
    //drawMatches(img1, kpts1, img2, kpts2, matches, output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //imshow("Matches", output);

    cout << "Match descriptors -end-" << endl;


    cout << "Compute fundamental matrix -start-" << endl;
    vector<Point2f> pts1(nb_matches);
    vector<Point2f> pts2(nb_matches);
    vector<int> queryIdxs( nb_matches ), trainIdxs( nb_matches );

    for( size_t i = 0; i < nb_matches; ++i )
    {
        queryIdxs[i] = matches[i].queryIdx;
        trainIdxs[i] = matches[i].trainIdx;
    }
    KeyPoint::convert(kpts1, pts1, trainIdxs);
    KeyPoint::convert(kpts2, pts2, queryIdxs);

    Mat F= findFundamentalMat(pts1, pts2, FM_RANSAC, 2, 0.99);
    //Mat F= findFundamentalMat(pts1, pts2, FM_LMEDS);
    print(F);
    cout << "Compute fundamental matrix -end-" << endl;


    /*
    cout << "Remove outliers -start-" << endl;
    vector<uchar> status(pts1.size());
    vector<Point2f> in_pts1(pts1.size());
    vector<Point2f> in_pts2(pts1.size());


    for (unsigned int i=0; i<status.size(); ++i)
    {
        //if (status[i])
        {
            in_pts1.push_back(pts1[i]);
            in_pts2.push_back(pts2[i]);
        }
    }
    cout << "Remove outliers -end-" << endl;



    cout << "Compute essential matrix -start-" << endl;
    Mat cam_matrix,distortion_coeff;
    Mat K1(3, 3, CV_64FC1);
    Mat K2(3, 3, CV_64FC1);
    cv::FileStorage r_fs;
    cv::FileStorage l_fs;

    r_fs.open("/home/gmanfred/Desktop/r_camera_calibration.yml",cv::FileStorage::READ);
    r_fs["camera_matrix"]>>K1;
    r_fs["distortion_coefficients"]>>distortion_coeff;
    l_fs.open("/home/gmanfred/Desktop/l_camera_calibration.yml",cv::FileStorage::READ);
    l_fs["camera_matrix"]>>K2;

    Mat E = K2.t() * F * K1; //according to HZ (9.12)
    print(E);
    cout << "Compute essential matrix -end-" << endl;


    cout << "Compute projection matrix -start-" << endl;
    Matx34d P1 = Matx34d(1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0);

    SVD svd(E);
    Matx33d W(0,-1,0,   //HZ 9.13
              1,0,0,
              0,0,1);
    Matx33d Winv(0,1,0,
                -1,0,0,
                 0,0,1);
    Mat_<double> R = svd.u * Mat(W) * svd.vt; //HZ 9.19
    Mat_<double> t = svd.u.col(2); //u3

    Matx34d P2 = Matx34d(R(0,0),    R(0,1), R(0,2), t(0),
                         R(1,0),    R(1,1), R(1,2), t(1),
                         R(2,0),    R(2,1), R(2,2), t(2));

    //check chirality
    Point3d tmp_hpts1(pts1[0].x, pts1[0].y, 1);
    Point3d tmp_hpts2(pts2[0].x, pts2[0].y, 1);
    Mat_<double> X = IterativeLinearLSTriangulation(tmp_hpts1,P1,tmp_hpts2,P2);

    if (X(2) < 0)
    {
        t = -t;
        P2 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
                     R(1,0),	R(1,1),	R(1,2),	t(1),
                     R(2,0),	R(2,1),	R(2,2), t(2));
        X = IterativeLinearLSTriangulation(tmp_hpts1,P1,tmp_hpts2,P2);

        if (X(2) < 0)
        {
            t = -t;
            R = svd.u * Mat(Winv) * svd.vt;
            P2 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
                         R(1,0),	R(1,1),	R(1,2),	t(1),
                         R(2,0),	R(2,1),	R(2,2), t(2));
            X = IterativeLinearLSTriangulation(tmp_hpts1,P1,tmp_hpts2,P2);

            if (X(2) < 0)
            {
                t = -t;
                P2 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
                             R(1,0),	R(1,1),	R(1,2),	t(1),
                             R(2,0),	R(2,1),	R(2,2), t(2));
                X = IterativeLinearLSTriangulation(tmp_hpts1,P1,tmp_hpts2,P2);
                if (X(2) < 0)
                {
                    cout << "Shit." << endl; exit(0);
                }
            }
        }
    }

    print(P2);

    cout << "Compute projection matrix -end-" << endl;


    cout << "Compute 3D points -start-" << endl;
    /*
    vector<Point3d> point_cloud;
    vector<double> depths;
    for(int i=0; i<pts1.size(); ++i)
    {
        Point3d hpt1(pts1[i].x, pts1[i].y, 1);
        Point3d hpt2(pts2[i].x, pts2[i].y, 1);
		Mat_<double> tmp1 = K1.t() * Mat_<double>(hpt1);
		hpt1 = tmp1.at<Point3d>(0);
        Mat_<double> tmp2 = K2.t() * Mat_<double>(hpt2);
		hpt2 = tmp2.at<Point3d>(0);
        Mat_<double> X = IterativeLinearLSTriangulation(hpt1, P1, hpt2, P2);
        point_cloud.push_back(Point3d(X(0),X(1),X(2)));
        //depths.push_back(X(2));
    }

    double minVal,maxVal;
	minMaxLoc(depths, &minVal, &maxVal);
	//Mat tmp(240,320,CV_8UC3); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
	//Mat tmp(640,480,CV_8UC3); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
	Mat tmp(640,480,CV_8UC3); cvtColor(img1, tmp, CV_BGR2HSV);
	for (unsigned int i=0; i<depths.size(); i++) {
		double _d = MAX(MIN((depths[i]-minVal)/(maxVal-minVal),1.0),0.0);
		circle(tmp, pts1[i], 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
	}
	cvtColor(tmp, tmp, CV_HSV2BGR);
	imshow("Depth Map", tmp);
	waitKey(0);
	destroyWindow("Depth Map");

    cout << "Compute 3D points -end-" << endl;


    cout << "Backprojecting 3D points -start-" << endl;
    /*
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
}
*/
