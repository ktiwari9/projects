#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Common.h"

bool FindCameraMatrices3D2D(cv::Mat_<double>& rvec,
                            cv::Mat_<double>& t,
                            cv::Mat_<double>& R,
                            std::vector<cv::Point3f> ppcloud,
                            std::vector<cv::Point2f> imgPoints,
                            bool use_gpu,
                            cv::Mat K, cv::Mat distortion_coeff);
