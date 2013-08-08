/*
 *  FindCameraMatrices.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <opencv2/core/core.hpp>

void FindCameraMatrices(const cv::Mat& K1,
						const cv::Mat& K2,
						const cv::Mat& K1inv,
						const cv::Mat& K2inv,
						const std::vector<cv::Point2f>& imgpts1,
						const std::vector<cv::Point2f>& imgpts2,
						std::vector<cv::Point2d>& imgpts1_good,
						std::vector<cv::Point2d>& imgpts2_good,
						cv::Matx34d& P,
						cv::Matx34d& P1
#ifdef __SFM__DEBUG__
						,const cv::Mat&, const cv::Mat&
#endif
						);
