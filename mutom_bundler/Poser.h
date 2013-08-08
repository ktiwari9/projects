#ifndef DEF_MUTOM_BUNDLER_POSER
#define DEF_MUTOM_BUNDLER_POSER

#include <opencv2/core/core.hpp>
#include <cstdio>

#include "5point.h"
#include "matrix.h"
#include "vector.h"
#include "triangulation.h"

#include "api.h"

class Poser
{
public:
  int estimate_pose_5point (std::vector<cv::KeyPoint> kpts1, std::vector<int> &idx1,
                             std::vector<cv::KeyPoint> kpts2, std::vector<int> &idx2,
                             double scale, cv::Mat Kin,
                             cv::Matx33d &R, cv::Matx31d &t);
  int estimate_pose_3point_cv (std::vector<cv::Point3d> p3d, std::vector<int> idx_3d,
                             std::vector<cv::KeyPoint> kpts, std::vector<int> idx_2d,
                             cv::Mat Kin,
                             cv::Matx33d &R, cv::Matx31d &t);
  int estimate_pose_3point_bd (std::vector<cv::Point3d> p3d, std::vector<int> idx_3d,
                             std::vector<cv::KeyPoint> kpts, std::vector<int> idx_2d,
                             cv::Mat Kin,
                             cv::Matx33d &R, cv::Matx31d &t);
  Poser ();
	Poser (const cv::Matx33d K);
	~Poser ();

private:
	double m_ransac_inliers;
	int m_ransac_rounds;
	float m_ransac_thresh;
};

#endif

