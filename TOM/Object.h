#ifndef TOM_OBJECT_H_
#define TOM_OBJECT_H_

#include <opencv2/opencv.hpp>

class Object {
 public:
  Object () {}
  void add_point3d (cv::Point3d);
  void add_point3d (cv::Point3d, cv::Mat descriptor);
  void add_descriptor (unsigned int index, cv::Mat descriptor);
 private:
  std::vector<cv::Point3d> p3d_;
  std::vector< std::vector<cv::Mat> > desc_;
};

#endif
