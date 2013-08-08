#ifndef DEF_MUTOM_BUNDLER_FRAME
#define DEF_MUTOM_BUNDLER_FRAME

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "api.h"

class Frame
{
public:
  int id();
  void clear ();
  void remove_from_neighbours(int frame_id);
  std::vector<cv::Point2d> get_matched_points(int id);
  int get_corr_3d_pts (int match_idx);
  void add_matches (int id, std::vector<int> idx);
  void add_triang_matches (int id, std::vector<int> idx);
  void add_3d_matches (int id, std::vector<int> idx);

	Frame();
	~Frame();

  int m_id;
  cv::Mat m_img;
	std::vector<cv::KeyPoint> m_kpts;
  cv::Mat m_descs;
	cv::Matx33d m_R;
  cv::Matx31d m_t;

  // Matches for each neighboor.
	std::map<int, std::vector<int> > m_matches;
	// For each neighboor frame, index of keypoints used to create a 3D points.
	std::map<int, std::vector<int> > m_triang_matches;
	// For each neighboor frame, index of 3D points seen for pose estimation or triangulated.
	std::map<int, std::vector<int> > m_3d_matches;
};


#endif // DEF_MUTOM_BUNDLER_FRAME
