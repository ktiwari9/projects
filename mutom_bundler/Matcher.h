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

	void match (std::vector<cv::KeyPoint> kpts1,	cv::Mat descs1,
							std::vector<cv::KeyPoint> kpts2,	cv::Mat descs2,
							std::vector<int> &pts_idx1, std::vector<int> &pts_idx2);

	void draw_matches ( const cv::Mat& img1, const cv::Mat& img2,
											std::vector<cv::KeyPoint> kpts1,
											std::vector<cv::KeyPoint> kpts2);

	cv::Mat get_image_matches (	const cv::Mat& img1, const cv::Mat& img2,
															std::vector<cv::KeyPoint> kpts1, std::vector<int> idx1,
															std::vector<cv::KeyPoint> kpts2, std::vector<int> idx2);
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
