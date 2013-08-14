#ifndef TOM_IMAGE_H_
#define TOM_IMAGE_H_

#include <opencv2/opencv.hpp>
#include "Common.h"

class Image {
 public:
  Image ();
  Image (cv::Mat image, int id);
  bool operator!=(const Image img);
  std::vector<cv::KeyPoint> kpts ();
  std::vector<cv::KeyPoint> kpts (int id);
  std::vector<cv::Point2f> p2d ();
  std::vector<cv::Point2f> p2d (int id);
  //std::vector<cv::Point3f> p3d (int id);
  std::vector<CloudPoint> p3d (int id);
  std::vector<cv::DMatch> matches (Image img);
  void set_image (cv::Mat image);
  void add_keypoints (std::vector<cv::KeyPoint> keypoints);
  void add_descriptors (cv::Mat descriptors);
  void set_matches (int id, std::vector<unsigned int> index);
  void set_point3d (std::vector<CloudPoint> p3d);
  void set_point3d (std::vector<cv::Point3f> p3d);
  void add_point3d (int id, std::vector<CloudPoint> p3d);
  // image_id : the Id of the corresponding image
  // index : the indices of the matched points in this image
  void add_match (int image_id, std::vector<unsigned int> index);
  void set_pose (cv::Matx34d P);
  // debug
  cv::Mat draw_matches (Image img);

// private:
  unsigned int id_;
  cv::Mat image_;
  std::vector<cv::KeyPoint> kpts_;
  std::map<int, std::vector<unsigned int> > matches_;
  cv::Mat desc_;
  //std::vector<cv::Point3f> p3d_;
  std::vector<CloudPoint> p3d_;
  cv::Matx34d P_;
};

#endif
