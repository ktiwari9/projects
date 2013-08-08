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

void Matcher::match ( Mat descs1, Mat descs2,
											vector<unsigned int> &pts_idx1,
											vector<unsigned int> &pts_idx2)
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

  for ( size_t i = 0; i < m_sym_matches.size(); i++ )
  {
  	//cout << m_sym_matches[i].distance << " / " << min_dist << endl;
    if ( m_sym_matches[i].distance <= 5*min_dist ) //GOOD FOR SURF
    //if ( m_sym_matches[i].distance <= 3*min_dist ) //GOOD FOR FAST
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
