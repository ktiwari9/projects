#ifndef DEF_MATCHER
#define DEF_MATCHER

#include <cstdio>

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Matcher
{
public:
	Matcher();
	~Matcher();

	void extract (const cv::Mat img,
			 					std::vector<cv::KeyPoint> &kpts,	cv::Mat &descs);

	void match (cv::Mat descs1, cv::Mat descs2,
							std::vector<unsigned int> &pts_idx1,
						  std::vector<unsigned int> &pts_idx2);

private:
	void symmetryTest (const std::vector<cv::DMatch>& matches1,
									    const std::vector<cv::DMatch>& matches2,
									    std::vector<cv::DMatch>& symMatches);

	cv::Ptr<cv::FeatureDetector> m_detector;
	cv::Ptr<cv::DescriptorExtractor> m_extractor;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;

	std::vector<cv::DMatch>	m_sym_matches;
};

#endif
