#ifndef DEF_MUTOM_BUNDLER_TRIANGULATER
#define DEF_MUTOM_BUNDLER_TRIANGULATER

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "matrix/matrix.h"
#include "matrix/vector.h"
#include "sfm-driver/sfm.h"
#include "imagelib/triangulate.h"
#include "imagelib/defines.h"

#include "api.h"

#define EPSILON 0.0001

class Triangulater
{
public:
  int triangulate_bundler (std::vector<cv::KeyPoint> kpts1, std::vector<int> &idx1,
                             std::vector<cv::KeyPoint> kpts2, std::vector<int> &idx2,
                             cv::Matx33d R0, cv::Matx31d t0,
                             cv::Matx33d R1, cv::Matx31d t1,
                             cv::Mat Kin, std::vector<cv::Point3d>& pointcloud);

  int triangulate_simple (std::vector<cv::KeyPoint> kpts1, std::vector<int> idx1,
                            std::vector<cv::KeyPoint> kpts2, std::vector<int> idx2,
                            cv::Matx33d R0, cv::Matx31d t0,
                            cv::Matx33d R1, cv::Matx31d t1,
                           cv::Mat Kinv, std::vector<cv::Point3d>& pointcloud);

  Triangulater ();
  ~Triangulater ();

private:
  // Functions for triangulate simple (roy shilkrot)
  cv::Mat_<double> linearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
                                           cv::Matx34d P,		//camera 1 matrix
                                           cv::Point3d u1,		//homogenous image point in 2nd camera
                                           cv::Matx34d P1		//camera 2 matrix
                                           );
  cv::Mat_<double> iterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
                                                    cv::Matx34d P,			//camera 1 matrix
                                                    cv::Point3d u1,			//homogeneous image point in 2nd camera
                                                    cv::Matx34d P1			//camera 2 matrix
                                                    );

  // Functions for triangulate bundler
  v3_t triangulate_robust(v2_t p, v2_t q,
                          camera_params_t c1, camera_params_t c2,
                          double &proj_error, bool &in_front, double &angle,
                          bool explicit_camera_centers);
  bool CheckCheirality(v3_t p, const camera_params_t &camera);
    /* Compute the angle between two rays */
  double ComputeRayAngle(v2_t p, v2_t q,
                         const camera_params_t &cam1,
                         const camera_params_t &cam2);
  v2_t UndistortNormalizedPoint(v2_t p, camera_params_t c);
  void GetIntrinsics (const camera_params_t &camera, double *K);
};

#endif // DEF_MUTOM_BUNDLER_TRIANGULATER
