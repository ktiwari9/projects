#include "Pipeline.h"

using namespace std;
using namespace cv;

Pipeline::Pipeline (double uc, double vc, double fu, double fv,
                    double a, double b, double c, double d, double e)
{
  K_ = (Mat_<double>(3, 3) << fu, 0, uc,
                               0, fv, vc,
                               0,  0,  1);
  dist_coeffs_ = (Mat_<double>(5, 1) << a, b, c, d, e);
  Kinv_ = K_.inv();
}

Pipeline::Pipeline (Mat K, Mat dist_coeffs): K_(K), dist_coeffs_ (dist_coeffs)
{
  Kinv_ = K_.inv();
}

void Pipeline::operator<<(const Image I) {
  Image image(I);
  Image working = curr_.back();
  Image last_keyframe = kf_.back();

  curr_.push (image);
  if (curr_.size() > MAX_CURRENT_FRAMES)
    curr_.pop(); // remove oldest frame
  extract (working);
  // last kf and last curr
  match (last_keyframe, working);
  compute_pose (working);

  if (last_keyframe != working) {
    Image former_working = curr_.front(); // on prend la frame precedente
    triangulate (last_keyframe, former_working);
    kf_.push_back (former_working);
  }
}

void Pipeline::extract (Image &img) {
  vector<KeyPoint> kpts;
  Mat desc;
  matcher_.extract (img.image_, kpts, desc);
  img.add_keypoints (kpts);
  img.add_descriptors (desc);
}

void Pipeline::match (Image &img1, Image &img2) {
  vector<unsigned int> idx1, idx2;
  // Check if Image had descriptors extracted
  if (!img1.desc_.data || !img2.desc_.data) {
    cerr << "Error : Pipeline.match : image(s) with no descriptors." << endl;
    return;
  }
  matcher_.match (img1.desc_, img2.desc_,
                  idx1, idx2);
  img1.add_match (img2.id_, idx1);
  img2.add_match (img1.id_, idx2);
}

void Pipeline::compute_pose (Image &img) {
  Matx34d P;
  int n = kf_.size();
  if ( n < 2 ) // not enough frames
    ;
  else if ( n == 2 ) // Npoints
    find_camera_matrix2D2D (img, kf_[n-1], P);
  else // PNP
    find_camera_matrix3D2D (img, kf_[n-1], P);

  img.set_pose (P);
}

void Pipeline::find_camera_matrix2D2D (Image img1, Image img2, Matx34d &P) {
  vector<CloudPoint> out_cloud;
  vector<KeyPoint> good_kpts1, good_kpts2;
  Matx34d P0 (1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
  vector<DMatch> match = img1.matches(img2);
  FindCameraMatrices2D2D (K_, Kinv_, dist_coeffs_,
                          img1.kpts(), img2.kpts(),
                          good_kpts1, good_kpts2,
                          P0, P,
                          match,
                          out_cloud);
  cout << "Fini" << endl;
}

void Pipeline::find_camera_matrix3D2D (Image img1, Image img2,
                                        Matx34d &P) {
  cv::Mat_<double> rvec, R, t;
  std::vector<cv::Point3f> p3d = img1.p3d(img2.id_);
  std::vector<cv::Point2f> p2d = img2.p2d(img1.id_);
  cout << p3d.size () << endl;
  cout << p2d.size () << endl;
  FindCameraMatrices3D2D (rvec, t, R,
                          p3d, p2d,
                          false, K_, dist_coeffs_);
  /*
  P = Matx34d(R(0,0), R(0,1), R(0,1), t(0,0),
              R(1,0), R(1,1), R(1,1), t(1,0),
              R(2,0), R(2,1), R(2,1), t(2,0));
  */
}

void Pipeline::triangulate (Image &img1, Image &img2) {

}

void find2D3Dcorrespondences (int working_view,
                            std::vector<cv::Point3f>& p3d,
                            std::vector<cv::Point2f>& p2d) {

                            }
