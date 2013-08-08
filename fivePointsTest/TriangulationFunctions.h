#ifndef DEF_TRIANGULATIONFUNCTIONS
#define DEF_TRIANGULATIONFUNCTIONS

#include <opencv2/core/core.hpp>
#include <vector>

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
								   cv::Matx34d P,		//camera 1 matrix
								   cv::Point3d u1,		//homogenous image point in 2nd camera
								   cv::Matx34d P1		//camera 2 matrix
								   );

#define EPSILON 0.0001
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
											cv::Matx34d P,			//camera 1 matrix
											cv::Point3d u1,			//homogenous image point in 2nd camera
											cv::Matx34d P1			//camera 2 matrix
											);

void TriangulatePoints(const std::vector<cv::Point2d>& pt_set1,
					   const std::vector<cv::Point2d>& pt_set2,
					   const cv::Mat& Kinv,
					   const cv::Matx34d& P,
					   const cv::Matx34d& P1,
					   std::vector<cv::Point3d>& pointcloud,
					   std::vector<cv::Point>& correspImg1Pt);

void solveChirality(cv::Point3d calib_hpt1, cv::Point3d calib_hpt2, cv::Mat_<double> Ra, cv::Mat_<double> Rb, cv::Mat_<double> t, cv::Matx34d &P);

#endif // DEF_TRIANGULATIONFUNCTIONS

