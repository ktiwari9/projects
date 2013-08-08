#include "Matcher.h"

using namespace std;
using namespace cv;

Matcher::Matcher()
{
  // init surf and sift modules.
	initModule_nonfree();

	m_detector = FeatureDetector::create("SURF");
  m_extractor = DescriptorExtractor::create("SIFT");
	m_matcher = DescriptorMatcher::create("BruteForce");
	//m_matcher = DescriptorMatcher::create("FlannBased");
	//m_matcher = DescriptorMatcher::create("BruteForce-Hamming");
}

Matcher::~Matcher()
{}

void Matcher::extract ( const Mat img,
			 									vector<KeyPoint> &kpts,	Mat &descs)
{
	// Detect keypoints
  m_detector->detect ( img, kpts );
	// Extract descriptor for each keypoint
  m_extractor->compute ( img, kpts, descs );
}

void Matcher::match ( vector<KeyPoint> kpts1,	Mat descs1,
											vector<KeyPoint> kpts2,	Mat descs2,
											vector<int> &pts_idx1, vector<int> &pts_idx2)
{
	m_sym_matches.clear();

  double min_dist = 1000; double dist = 0;
  vector< DMatch > matches1;
  vector< DMatch > matches2;

  m_matcher->match ( descs1, descs2, matches1 );
  m_matcher->match ( descs2, descs1, matches2 );

  // Remove non symmetrical matches
  symmetryTest (matches1, matches2, m_sym_matches);

	for ( size_t i = 0; i < m_sym_matches.size(); i++ )
	{
	    dist = m_sym_matches[i].distance;
	    if ( dist < min_dist && dist != 0 ) min_dist = dist; // Take care of the min_dist == 0 case.
	}
	cout << "Min distance = " << min_dist << endl;

  for ( size_t i = 0; i < m_sym_matches.size(); i++ )
  {
    if ( m_sym_matches[i].distance <= 5*min_dist ) //GOOD FOR SURF
    {
      pts_idx1.push_back ( m_sym_matches[i].queryIdx );
      pts_idx2.push_back ( m_sym_matches[i].trainIdx );
    }
  }
}

void Matcher::symmetryTest (const vector<DMatch>& matches1,
												    const vector<DMatch>& matches2,
												    vector<DMatch>& symMatches)
{
    // for all matches image 1 -> image 2
    for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();
    		 matchIterator1!= matches1.end();
    		 ++matchIterator1) {
       // for all matches image 2 -> image 1
       for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();
            matchIterator2!= matches2.end();
            ++matchIterator2) {
           // Match symmetry test
           if ((*matchIterator1).queryIdx ==
               (*matchIterator2).trainIdx &&
               (*matchIterator2).queryIdx ==
               (*matchIterator1).trainIdx) {
               // add symmetrical match
                 symMatches.push_back(
                   DMatch((*matchIterator1).queryIdx,
                             (*matchIterator1).trainIdx,
                             (*matchIterator1).distance));
                 break; // next match in image 1 -> image 2
           }
       }
    }
}

void Matcher::draw_matches (const Mat& img1, const Mat& img2,
														vector<KeyPoint> kpts1, vector<KeyPoint> kpts2)
{
	Mat outImg;
	namedWindow ("Matches Debug");
	drawMatches ( img1, kpts1, img2, kpts2, m_sym_matches, outImg,
								Scalar::all(-1), Scalar::all(-1), vector<char>(),
								DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow ("Matches Debug", outImg);
	waitKey (1);
}

Mat Matcher::get_image_matches (const Mat& img1, const Mat& img2,
																vector<KeyPoint> kpts1, vector<int> idx1,
																vector<KeyPoint> kpts2, vector<int> idx2)
{
	Mat outImg;
	int n = idx1.size();
	std::vector<cv::DMatch>	matches;
	for (size_t i=0; i<n; ++i)
	{
    matches.push_back ( DMatch(idx1[i], idx2[i], 1.0));
	}

	drawMatches ( img1, kpts1, img2, kpts2, matches, outImg,
								Scalar::all(-1), Scalar::all(-1), vector<char>(),
								DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	return outImg;
}
