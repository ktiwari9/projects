#ifndef DEF_MUTOM_BUNDLER_API
#define DEF_MUTOM_BUNDLER_API

#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "5point/5point.h"
#include "matrix/matrix.h"
#include "matrix/vector.h"
#include "imagelib/fmatrix.h"
#include "imagelib/triangulate.h"
#include "sfm-driver/sfm.h"

#include "api_struct.h"
#include "tools.h"

void cv2bd ( std::vector<cv::Point2d> in, v2_t* out);
void cv2bd ( std::vector<cv::Point3d> in, v3_t* out);
void bd2cv ( int n, v2_t* in, std::vector<cv::Point2f> &out);
void bd2cv ( int n, v3_t* in, std::vector<cv::Point3f> &out);
void bd2cv ( int n, v2_t* in, std::vector<cv::Point2d> &out);
void bd2cv ( int n, v3_t* in, std::vector<cv::Point3d> &out);
void mat_cv2bd (cv::Mat in_K, double* out_K);
void mat_cv2bd (cv::Matx33d in, double* out);
void mat_bd2cv (double* out_K, cv::Matx33d& in_K);
void mat_cv2bd (cv::Matx31d in, double* out);
std::vector<cv::Point2d> idx2pts (std::vector<cv::KeyPoint> kpts,
                                  std::vector<int> idx);
std::vector<cv::Point3d> idx2pts (std::vector<cv::Point3d> p3d,
                                  std::vector<int> idx);
camera_params_t camera_params_new (double* R, double* t, double* K);
std::vector<cv::Point2f> d2f(std::vector<cv::Point2d> in);
std::vector<cv::Point3f> d2f(std::vector<cv::Point3d> in);
#endif // DEF_MUTOM_BUNDLER_API

