#ifndef DEF_MUTOM_BUNDLER_BUNDLE
#define DEF_MUTOM_BUNDLER_BUNDLE

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Frame.h"
#include "Pipeline.h"

#define MATCH_THRESH 200
#define POSE_TRESH 0
#define LOCAL_BUNDLE_SIZE 4

class Bundle
{
public :
  void run (cv::Mat img);
  char not_enough_matches (Frame &frm);
  char pose_too_uncertain(Frame &frm);
  void compute_p3d(Frame &frm1, Frame &frm2);
  void add_frame_to_kf(Frame frm);
  void local_bundle (int n);

  Bundle(cv::Mat K);
  ~Bundle();

private :
  Pipeline pipe;

  char m_init;

  int m_n_frames;
  int m_n_kf;
  std::vector<cv::Point3d> m_p3d;
  Frame m_current_frame;
  Frame m_former_frame;
  std::vector<Frame> m_kf;
};

#endif // DEF_MUTOM_BUNDLER_BUNDLE
