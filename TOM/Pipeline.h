#ifndef TOM_PIPELINE_H_
#define TOM_PIPELINE_H_

#include <queue>

#include "Image.h"
#include "Object.h"

#include "Matcher.h"
#include "FindCameraMatrices2D2D.h"
#include "FindCameraMatrices3D2D.h"

#define MAX_CURRENT_FRAMES 2 //TODO change this to 3 and add nister
// last keyframe == kf_.back()
// last current frame == curr_.back()

class Pipeline {
 public:
  Pipeline (double uc, double vc, double fu, double fv,
            double a, double b, double c, double d, double e);
  Pipeline (cv::Mat K, cv::Mat dist_coeffs);
  void operator<<(const Image img);
  void extract (Image &img);
  void match (Image &img1, Image &img2);
  void compute_pose (Image &img);
  void triangulate (Image &img1, Image &img2);

  Object build_object ();
  void save_object (std::string path);
// private:
  void find_camera_matrix2D2D (Image img1, Image img2, cv::Matx34d &P);
  void find_camera_matrix3D2D (Image img1, Image img2, cv::Matx34d &P);
  void find2D3Dcorrespondences (int working_view,
                                std::vector<cv::Point3f>& p3d,
                                std::vector<cv::Point2f>& p2d);

  Matcher matcher_;
  std::vector<Image> kf_; // keyframe
  std::queue<Image> curr_; // current frames, discardable
  cv::Mat covis_; // covisibility matrix

  cv::Mat K_, Kinv_;
  cv::Mat dist_coeffs_;
};

#endif
