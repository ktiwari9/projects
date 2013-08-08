#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/visualization/cloud_viewer.h>

#include "matrix/vector.h"
#include "sfm-driver/sfm.h"
#include "BundleAdd.h"

#include "api.h"
#include "Pipeline.h"

using namespace std;
using namespace cv;

void test_pipeline(char* filename1, char* filename2);
void test_solo(char* filename1, char* filename2);

int main (int argc, char** argv)
{
  argc = 3;
  argv = (char**) malloc (100*sizeof (char));
  argv[1] = "/home/gmanfred/data/rightRect0000.jpg";
  argv[2] = "/home/gmanfred/data/leftRect0000.jpg";
//  argv[1] = "/home/gmanfred/data/frame0000.jpg";
//  argv[2] = "/home/gmanfred/data/frame0003.jpg";
//  argv[1] = "/home/gmanfred/data/et000.jpg";
//  argv[2] = "/home/gmanfred/data/et003.jpg";

  if (argc != 3)
  {
    cout << "Usage : mutom_bundler image1 image2";
    exit (-1);
  }

  //test_solo(argv[1], argv[2]);
  test_pipeline(argv[1], argv[2]);

  return 0;
}

void test_pipeline(char* filename1, char* filename2)
{
  Pipeline pipe;

  cout << "---------------- Opening images ----------------" << endl;
  Mat img1 = imread (filename1);
  Mat img2 = imread (filename2);
  Mat img3 = imread (filename2);

  Frame frm1, frm2, frm3;
  frm1.m_img = img1;
  frm2.m_img = img2;
  frm3.m_img = img3;
  frm1.m_id = 1;
  frm2.m_id = 2;
  frm3.m_id = 3;

  cout << "---------------- Loading camera parameters files ----------------" << endl;
  Mat K (3, 3, CV_64FC1);
  cv::FileStorage r_fs;
  r_fs.open ("/home/gmanfred/.ros/camera_info/r_camera_calibration.yml", cv::FileStorage::READ);
  r_fs["camera_matrix"]>>K;
  pipe.setK (K);

  //---------------- Matching features ----------------
  pipe.extract_features (frm1);
  pipe.extract_features (frm2);
  pipe.match_frames (frm1, frm2);

  //---------------- Estimating pose ----------------
  frm1.m_R = Matx33d (1, 0, 0,
                      0, 1, 0,
                      0, 0, 1);
  frm1.m_t = Matx31d (0, 0, 0);

  pipe.estimate_pose_init (frm1, frm2);

  printMatx33(frm1.m_R);
  cout << frm1.m_t(0,0) << " " << frm1.m_t(1,0) << " " << frm1.m_t(2,0) << endl;
  cout << endl;
  printMatx33(frm2.m_R);
  cout << frm2.m_t(0,0) << " " << frm2.m_t(1,0) << " " << frm2.m_t(2,0) << endl;

  //---------------- Debug ------------------------
  frm2.m_R = Matx33d (1, 0, 0,
                      0, 1, 0,
                      0, 0, 1);
  frm2.m_t = Matx31d (1, 0, 0);
  //---------------- Triangulating ----------------
  vector<Point3d> p3d;
  pipe.triangulate_frames (frm1, frm2, p3d);
  //pipe.show_matches_triang(frm1, frm2);

  //---------------- Refining ----------------
  pipe.refine_frames (frm1, frm2, p3d);

  //----------------- Estimating pose from estimated 3d points ------------
  pipe.extract_features (frm3);
  pipe.match_frames (frm1, frm3);
  pipe.estimate_pose_run (frm1, frm3, p3d);

  printMatx33(frm3.m_R);
  cout << frm3.m_t(0,0) << " " << frm3.m_t(1,0) << " " << frm3.m_t(2,0) << endl;

  //print_to_file(p3d);
}

void test_solo(char* filename1, char* filename2)
{
  Matcher matcher;

  cout << "---------------- Opening images ----------------" << endl;
  Mat img1 = imread (filename1);
  Mat img2 = imread (filename2);


  cout << "---------------- Loading camera parameters files ----------------" << endl;
  Mat K(3, 3, CV_64FC1);
  cv::FileStorage r_fs;
  r_fs.open("/home/gmanfred/.ros/camera_info/r_camera_calibration.yml",cv::FileStorage::READ);
  r_fs["camera_matrix"]>>K;


  cout << "---------------- Matching features ----------------" << endl;
  vector<KeyPoint> kpts1, kpts2;
  Mat descs1, descs2;
  vector<int> idx1, idx2;
  matcher.extract (img1, kpts1, descs1);
  matcher.extract (img2, kpts2, descs2);
  matcher.match (kpts1, descs1, kpts2, descs2,
                  idx1, idx2);
  cout << "Matched " << idx1.size() << " features." << endl;


  cout << "---------------- Estimating pose ----------------" << endl;
	int n = idx1.size ();
	int n_in = 0;
	v2_t* vp1 = (v2_t*) malloc (n*sizeof (v2_t));
	v2_t* vp2 = (v2_t*) malloc (n*sizeof (v2_t));
	double* inliers = (double*) malloc (n*sizeof(double));
	double *Kd = (double*) malloc (9*sizeof(double));
	double *R = (double*) malloc (9*sizeof(double));
	double *t = (double*) malloc (3*sizeof(double));

  cv2bd ( idx2pts (kpts1, idx1), vp1 );
  cv2bd ( idx2pts (kpts2, idx2), vp2 );
  mat_cv2bd(K, Kd);

  //print_to_file("robust_solo.txt", n, vp2);
  n_in = compute_pose_ransac(n, vp1, vp2, Kd, Kd, 0.5, 400, R, t, inliers);
  printMatx33(R);
  cout << t[0] << " " << t[1] << " " << t[2] << endl;

  cout << "---------------- Selecting inliers ----------------" << endl;
  v2_t* in_vp1 = (v2_t*) malloc (n_in*sizeof (v2_t));
	v2_t* in_vp2 = (v2_t*) malloc (n_in*sizeof (v2_t));
  selectInliers (n_in, vp1, inliers, in_vp1);
  selectInliers (n_in, vp2, inliers, in_vp2);

  cout << n_in << "/" << n << " inliers." << endl;
  //for(int i=0; i<n; ++i)
  //  cout << i << " " << Vx(in_vp2[i]) << " " << Vy(in_vp2[i]) << " " << endl;

  cout << "---------------- Triangulating ----------------" << endl;
  bool in_front = true;
  double proj_error = 0.0;
  double angle = 0.0;
  double R0[9] = {1,0,0,0,1,0,0,0,1};
  double t0[3] = {0,0,0};

  camera_params_t c0 = camera_params_new(R0, t0, Kd);
  camera_params_t c1 = camera_params_new(R, t, Kd);
  v3_t* vp3d = (v3_t*) malloc (n*sizeof (v3_t));

  for (int i=0; i<n_in; ++i)
  {
//    cout << "(" << Vx(vp1[i]) << "," << Vy(vp1[i]) << ")";
//    cout << "(" << Vx(vp2[i]) << "," << Vy(vp2[i]) << ")" << endl;
    vp3d[i] = triangulate_bundler (in_vp1[i], in_vp2[i],
                                    c0,  c1,
                                    proj_error, in_front, angle,
                                    true);
  }

  camera_refine(n_in, vp3d, in_vp2, &c1, 0, 0);

  print_to_file("robust_solo.txt", n_in, vp3d);
}
