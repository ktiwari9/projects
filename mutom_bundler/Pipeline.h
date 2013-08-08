#ifndef DEF_MUTOM_BUNDLER_PIPELINE
#define DEF_MUTOM_BUNDLER_PIPELINE

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "Matcher.h"
#include "Poser.h"
#include "Triangulater.h"

class Pipeline
{
public :
  int extract_features (Frame &frm1);
  int match_frames (Frame &frame1, Frame &frame2);
  void get_pose_matches(Frame &frm1, Frame &frm2,
                         std::vector<int> &out_pose_2d,
                         std::vector<int> &out_pose_3d);
  void get_triang_matches(Frame &frm1, Frame &frm2,
                         std::vector<int> &out_triang1,
                         std::vector<int> &out_triang2);
  double estimate_pose_init (Frame &frame1, Frame &frame2);
  double estimate_pose_run (Frame &frame1, Frame &frame2, std::vector<cv::Point3d> p3d);
  std::vector<cv::Point3d> local_bundle_adjustement (std::vector<Frame> &frames,
                                                     std::vector<cv::Point3d> p3d);
  int triangulate_frames (Frame &frame1, Frame &frame2,
                          std::vector<cv::Point3d> &p3d);
  void refine_frames (Frame frm1, Frame &frm2, std::vector<cv::Point3d> p3d);
  void setK(cv::Mat K);

  void show_matches(Frame frm1, Frame frm2);
  void show_matches_triang(Frame frm1, Frame frm2);
  Pipeline();
  ~Pipeline();

private :
  // Debug vars
  clock_t m_start;
  clock_t m_ends;

  cv::Mat m_K;
  cv::Mat m_Kinv;

  Matcher M;
  Poser P;
  Triangulater T;
};

#endif // DEF_MUTOM_BUNDLER_PIPELINE

