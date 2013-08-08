#ifndef DEF_MATCHER
#define DEF_MATCHER

#include "assert.h"

#include "libmv/multiview/five_point.h"
#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

class Matcher
{
public:
    /**
    *   Detect features, describe them, match them and returns only the
    *   best matches.
    */
    Matcher(char* detectorType, char* descriptorType, char* matchingType);
    std::vector< cv::DMatch > match(cv::Mat img1, cv::Mat img);
    int getNbMatches();
    void calibratePoints(cv::Mat K);
    void calibratePoints(cv::Mat K1, cv::Mat K2);

    cv::Mat _calib_hpts1;
    cv::Mat _calib_hpts2;

protected:
    void goodKptsToPts();

    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::Ptr<cv::DescriptorMatcher> _matcher;

    std::vector<cv::KeyPoint> _kpts1;
    std::vector<cv::KeyPoint> _kpts2;
    cv::Mat _descs1;
    cv::Mat _descs2;
    std::vector<cv::DMatch> _matches;
    std::vector<cv::DMatch> _good_matches;
    int _nb_matches;

    std::vector<cv::Point2f> _pts1;
    std::vector<cv::Point2f> _pts2;
    cv::Mat_<double> _hpts1;
    cv::Mat_<double> _hpts2;
};


#endif // DEF_MATCHER

