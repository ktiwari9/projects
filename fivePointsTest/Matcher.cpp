#include "Matcher.h"

using namespace cv;
using namespace std;

Matcher::Matcher(char* detectorType, char* descriptorType, char* matchingType)
{
    _detector = FeatureDetector::create(detectorType);
    _descriptor = DescriptorExtractor::create(descriptorType);
    _matcher = DescriptorMatcher::create(matchingType);

    _nb_matches = 0;
}

vector< DMatch > Matcher::match(Mat img1, Mat img2)
{
    /// Keypoints detection
    _detector->detect(img1, _kpts1);
    _detector->detect(img2, _kpts2);
    assert((!_kpts1.empty() || !_kpts2.empty()) && "No keypoint found");


    /// Keypoints description
    _descriptor->compute(img1, _kpts1, _descs1);
    _descriptor->compute(img2, _kpts2, _descs2);
    assert( (!_descs1.empty() || !_descs2.empty() ) && "Descriptor vectors null");

    /// Descriptors matching
    _matcher->match(_descs1, _descs2, _matches);

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < _descs1.rows; i++ )
    {
        double dist = _matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Keep only "good" matches (i.e. whose distance is less than 3*min_dist )
    for( int i = 0; i < _descs1.rows; i++ )
    {
        if( _matches[i].distance < 3*min_dist )
        {
            _good_matches.push_back(_matches[i]);
        }
    }
    _nb_matches = _good_matches.size();
    assert( _nb_matches != 0 && "No matches found");

    return _good_matches;
}

void Matcher::calibratePoints(Mat K)
{
    goodKptsToPts();

    Mat invK(3, 3, CV_64FC1);
    invert(K, invK);

    _hpts1 = Mat(3, _pts1.size(), CV_64FC1);
    _hpts2 = Mat(3, _pts2.size(), CV_64FC1);
    for(int i=0; i<_pts1.size(); ++i)
    {
        _hpts1(0, i) = _pts1[i].x;
        _hpts1(1, i) = _pts1[i].y;
        _hpts1(2, i) = 1;
        _hpts2(0, i) = _pts2[i].x;
        _hpts2(1, i) = _pts2[i].y;
        _hpts2(2, i) = 1;
    }

    _calib_hpts1 = invK*Mat(_hpts1);
    _calib_hpts2 = invK*Mat(_hpts2);
}

void Matcher::calibratePoints(Mat K1, Mat K2)
{
    Mat invK1(3, 3, CV_64FC1);
    Mat invK2(3, 3, CV_64FC1);

    goodKptsToPts();

    invert(K1, invK1);
    invert(K2, invK2);

    _hpts1 = Mat(3, _pts1.size(), CV_64FC1);
    _hpts2 = Mat(3, _pts2.size(), CV_64FC1);
    for(int i=0; i<_pts1.size(); ++i)
    {
        _hpts1(0, i) = _pts1[i].x;
        _hpts1(1, i) = _pts1[i].y;
        _hpts1(2, i) = 1;
        _hpts2(0, i) = _pts2[i].x;
        _hpts2(1, i) = _pts2[i].y;
        _hpts2(2, i) = 1;
    }

    _calib_hpts1 = invK1*_hpts1;
    _calib_hpts2 = invK2*_hpts2;
}

int Matcher::getNbMatches()
{
    return _nb_matches;
}

void Matcher::goodKptsToPts()
{
    _pts1.resize(_nb_matches);
    _pts2.resize(_nb_matches);
    vector<int> queryIdxs( _nb_matches ), trainIdxs( _nb_matches );

    for( size_t i = 0; i < _nb_matches; ++i )
    {
        queryIdxs[i] = _good_matches[i].queryIdx;
        trainIdxs[i] = _good_matches[i].trainIdx;
    }
    KeyPoint::convert(_kpts1, _pts1, trainIdxs);
    KeyPoint::convert(_kpts2, _pts2, queryIdxs);
}
