#include "Bundle.h"

using namespace std;

void Bundle::run (cv::Mat img)
{
  m_current_frame.clear ();
  m_current_frame.m_img = img;
  m_current_frame.m_id = m_n_frames;

  // Match frame with last keyframe
  if (not_enough_matches (m_current_frame)
      || pose_too_uncertain (m_current_frame))
  {
    compute_p3d (m_kf[m_n_kf-1], m_former_frame);
    add_frame_to_kf (m_former_frame);
    local_bundle (LOCAL_BUNDLE_SIZE);
  }
  else
  {
    if (m_n_kf < 1)
      m_kf[m_n_kf-1].remove_from_neighbours(m_former_frame.m_id);
  }

  m_former_frame = m_current_frame;
  ++m_n_frames;
}

char Bundle::not_enough_matches (Frame &frm)
{
  // Safety check to ensure that a keyframe already exists
  if (m_n_kf < 1)
  {
    pipe.extract_features (frm);
    return 0;
  }

  if (pipe.match_frames (m_kf[m_n_kf-1], frm) < MATCH_THRESH)
    return 1;
  else
    return 0;
}

char Bundle::pose_too_uncertain (Frame &frm)
{
  // Safety check to ensure that a keyframe already exists
  if (m_n_kf < 1)
  {
    cout << "Error: Bundle::pose_too_uncertain: this shouldn't happen" << endl;
    return 0;
  }

  double uncertainty = 0.0;
  if (m_init)
  {
    uncertainty = pipe.estimate_pose_init (m_kf[m_n_kf-1], frm);
    m_init = 0;
  }
  else
    uncertainty = pipe.estimate_pose_run (m_kf[m_n_kf-1], frm, m_p3d);

  if (uncertainty < POSE_TRESH)
    return 1;
  else
    return 0;
}

void Bundle::compute_p3d(Frame &frm1, Frame &frm2)
{
  pipe.triangulate_frames (frm1, frm2, m_p3d);
}

void Bundle::add_frame_to_kf (Frame frm)
{
  m_kf.push_back(frm);
  ++m_n_kf;
}

// Apply bundle adjustement on n last keyframes.
void Bundle::local_bundle (int n)
{
  vector<Frame> to_bundle;
  // Bundle adjust n keyframes if enough, else use all keyframes
  for (int i=0; i< min(n, m_n_kf); ++i)
    to_bundle.push_back(m_kf[m_n_kf - 1 - n]);

  pipe.local_bundle_adjustement (to_bundle, m_p3d);
}

Bundle::Bundle (cv::Mat K)
{
  m_init = 1;
  m_n_frames = 0;

  pipe.setK(K);
}

Bundle::~Bundle ()
{}
