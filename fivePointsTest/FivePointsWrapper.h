#ifndef DEF_FIVEPOINTS
#define DEF_FIVEPOINTS

#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "libmv/multiview/five_point.h"
#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "auxFunctions.h"

class FivePoints
{
public:
    FivePoints();
    void compute(cv::Mat calib_pts1, cv::Mat calib_pts2, cv::Mat_<double> &Ra, cv::Mat_<double> &Rb, cv::Mat_<double> &t);
    //cv::Matx33d libmvFivePoints(cv::Mat_<double> calib_hpts1, cv::Mat_<double> calib_hpts2);
    std::vector<cv::Matx33d> libmvFivePoints(cv::Mat_<double> calib_hpts1, cv::Mat_<double> calib_hpts2);
    cv::Matx33d getBestCandidate(std::vector< cv::Matx33d > candidates, cv::Mat_<double> HQ1, cv::Mat_<double> HQ2);
    //void essentialToRotationTranslation(cv::Mat_<double> E, cv::Mat_<double> &Ra, cv::Mat_<double> &Rb, cv::Mat_<double> &t);
    void essentialToRotationTranslation(cv::Matx33d E, cv::Mat_<double> &Ra, cv::Mat_<double> &Rb, cv::Mat_<double> &t);
    double getMinError();

    //libmv::vector<libmv::Mat3> _best_E;
    //cv::Mat_<double> _best_E;
    cv::Matx33d _best_E;

protected:
    int _num_iter;
    double _min_error;
};

#endif // DEF_FIVEPOINTSWRAPPER

