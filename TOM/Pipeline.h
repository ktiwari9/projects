#ifndef TOM_PIPELINE_H_
#define TOM_PIPELINE_H_

#include <queue>

#include "Image.h"
#include "Object.h"

#include "Matcher.h"
#include "FindCameraMatrices2D2D.h"
#include "FindCameraMatrices3D2D.h"
#include "Triangulation.h"
#include "BundleAdjuster.h"

#define MAX_CURRENT_FRAMES 2 //TODO change this to 3 and add nister
// last keyframe == kf_.back()
// last current frame == curr_.back()

class Pipeline {
 public:
  Pipeline (double uc, double vc, double fu, double fv,
            double a, double b, double c, double d, double e);
  Pipeline (cv::Mat K, cv::Mat dist_coeffs);

  void sfm_on_dir (char* path);
  void sfm_two_images (cv::Mat img1, cv::Mat img2);

  void operator<<(const Image img);
  void extract (Image &img);
  void match (Image &img1, Image &img2);
  void compute_pose (Image &img1, Image &img2);
  void triangulate (Image &img1, Image &img2);
  void adjust_bundle ();

  Object build_object ();
  void save_object (std::string path);

  void find_camera_matrix2D2D (Image &img1, Image &img2, cv::Matx34d &P);
  void find_camera_matrix3D2D (Image &img1, Image &img2, cv::Matx34d &P);

// private:
  std::vector<CloudPoint> merge_pointcloud (std::vector< std::vector<CloudPoint> > clouds);

  BundleAdjuster BA;
  Matcher matcher_;
  std::vector<Image> kf_; // keyframe
  std::queue<Image> curr_; // current frames, discardable
  cv::Mat covis_; // covisibility matrix

  cv::Mat K_, Kinv_;
  cv::Mat dist_coeffs_;

  unsigned int curr_id_;
};

#endif
