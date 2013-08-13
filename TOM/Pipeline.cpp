#include "Pipeline.h"

using namespace std;
using namespace cv;

Pipeline::Pipeline (double uc, double vc, double fu, double fv,
                    double a, double b, double c, double d, double e) {
  K_ = (Mat_<double>(3, 3) << fu, 0, uc,
                               0, fv, vc,
                               0,  0,  1);
  dist_coeffs_ = (Mat_<double>(5, 1) << a, b, c, d, e);
  Kinv_ = K_.inv();
  curr_id_ = 0;
}

Pipeline::Pipeline (Mat K, Mat dist_coeffs): K_(K), dist_coeffs_ (dist_coeffs)
{
  Kinv_ = K_.inv();
  curr_id_ = 0;
}

void Pipeline::sfm_on_dir (char *path) {
  std::vector<cv::Mat> images;
  std::vector<std::string> images_names;
  double downscale_factor = 1.0;
  open_imgs_dir(path, images, images_names, downscale_factor);
}

void Pipeline::sfm_two_images (Mat img1, Mat img2) {
  Image image1(img1, curr_id_);  ++curr_id_;
  Image image2(img2, curr_id_);  ++curr_id_;

  extract (image1);
  extract (image2);
  match (image1, image2);
  if (image1.id_ == 0) {
    Matx34d P0 (1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0);
    image1.set_pose (P0);
  }
  compute_pose (image1, image2);
  triangulate (image1, image2);
  kf_.push_back (image1);
  kf_.push_back (image2);

  int count = 0;
  for (size_t i=0; i<image1.p3d_.size(); ++i) {
    CloudPoint tmp = image1.p3d_[i];
    if (tmp.pt.x != 0 || tmp.pt.y != 0 || tmp.pt.z != 0)
      ++count;
  }

  cout << "Got " << count << " points 3D." << endl;
}

void Pipeline::operator<<(const Image I) {
  Image image(I);
  curr_.push (image);

  Image working = curr_.back();
  Image last_keyframe = kf_.back();

  if (curr_.size() > MAX_CURRENT_FRAMES)
    curr_.pop(); // remove oldest frame
  extract (working);
  // last kf and last curr
  match (last_keyframe, working);
  compute_pose (last_keyframe, working);

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

void Pipeline::compute_pose (Image img1, Image &img2) {
  Matx34d P;
  int n = kf_.size();
  if ( n <= 2 ) {// Npoints
    find_camera_matrix2D2D (img1, img2, P);
    cout << "Found pose with 2D2D : " << P << endl;
  }
  else {// PNP
    find_camera_matrix3D2D (img1, img2, P);
    cout << "Found pose with 3D2D : " << P << endl;
  }

  img2.set_pose (P);
}

void Pipeline::triangulate (Image &img1, Image &img2) {
  vector<KeyPoint> correspImg1Pt;
  vector<CloudPoint> pointcloud;
  cout << img1.kpts(img2.id_).size () << endl;
  TriangulatePoints(img1.kpts(img2.id_),
                    img2.kpts(img1.id_),
                    K_, Kinv_, dist_coeffs_,
                    img1.P_, img2.P_,
                    pointcloud,
                    correspImg1Pt);
  vector<Point2f> p2d1 = img1.p2d (img2.id_);
  vector<Point2f> p2d2 = img2.p2d (img1.id_);
  vector<DMatch> matches = img1.matches (img2); // 1->2 matches
  for (size_t i=0; i<pointcloud.size(); ++i) {
    pointcloud[i].imgpt_for_img.push_back (matches[i].queryIdx);
    pointcloud[i].imgpt_for_img.push_back (matches[i].trainIdx);
    //cout << pointcloud[i].imgpt_for_img.size () << endl;
  }

  img1.add_point3d (img2.id_, pointcloud);
  img2.add_point3d (img1.id_, pointcloud);
}

void Pipeline::adjust_bundle () {
  map<int, Matx34d> Pmats;
  vector< vector<KeyPoint> > imgpts;
  vector< vector<CloudPoint> > pointcloud;
  for (size_t i=0; i< kf_.size(); ++i) {
    Pmats.insert (std::pair<int, Matx34d>(kf_[i].id_, kf_[i].P_));
    imgpts.push_back (kf_[i].kpts_);
    pointcloud.push_back (kf_[i].p3d_);
  }
  vector<CloudPoint> cloud = merge_pointcloud (pointcloud);
  BA.adjustBundle(cloud, K_, imgpts, Pmats);

  cout << Pmats[1] << endl;
  cout << endl;
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
}

void Pipeline::find_camera_matrix3D2D (Image img1, Image img2,
                                        Matx34d &P) {
  cv::Mat_<double> rvec, R, t;
  std::vector<cv::Point3f> p3d;
  for (size_t i=0; i<p3d.size(); ++i)
    p3d.push_back (img1.p3d(img2.id_)[i].pt);
  std::vector<cv::Point2f> p2d = img2.p2d(img1.id_);
  cout << p3d.size () << endl;
  cout << p2d.size () << endl;

  bool success =
  FindCameraMatrices3D2D (rvec, t, R,
                          p3d, p2d,
                          false, K_, dist_coeffs_);

  if (success)
    P = Matx34d(R(0,0), R(0,1), R(0,2), t(0,0),
                R(1,0), R(1,1), R(1,2), t(1,0),
                R(2,0), R(2,1), R(2,2), t(2,0));
  else
    P = Matx34d(1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);
}

vector<CloudPoint> Pipeline::merge_pointcloud (vector< vector<CloudPoint> > clouds) {
  vector<CloudPoint> res;
  int count = 0;
  for (size_t i=0; i<clouds.size (); ++i) {
    for (size_t j=0; j<clouds[i].size (); ++j) {
      if ( !is_in (clouds[i][j], res) )
        res.push_back (clouds[i][j]);
    }
  }
  return res;
}












