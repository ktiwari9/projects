#include "Image.h"

using namespace std;
using namespace cv;

Image::Image () {
  cout << "Warning : Image constructor : empty Image instance created" << endl;
}

Image::Image (Mat image, int id): id_(id) {
  set_image (image);
}

bool Image::operator!=(const Image img) {
  if (matches_[img.id_].size () < 400)
    return true;
  return false;
}

std::vector<cv::KeyPoint> Image::kpts () {
  vector<KeyPoint> res;
  for (int i=0; i<kpts_.size(); ++i)
    res.push_back (kpts_[i]);
  return res;
}

std::vector<cv::KeyPoint> Image::kpts (int id) {
  vector<KeyPoint> res;
  for (int i=0; i<matches_[id].size(); ++i)
    res.push_back (kpts_[matches_[id][i]]);
  return res;
}

vector<Point2f> Image::p2d () {
  vector<Point2f> res;
  for (int i=0; i<kpts_.size(); ++i)
    res.push_back (kpts_[i].pt);
  return res;
}

vector<Point2f> Image::p2d (int id) {
  vector<Point2f> res;
  for (int i=0; i<matches_[id].size(); ++i)
    res.push_back (kpts_[matches_[id][i]].pt);
  return res;
}

//vector<Point3f> Image::p3d (int id) {
vector<CloudPoint> Image::p3d (int id) {
  vector<CloudPoint> res;
  for (int i=0; i<matches_[id].size(); ++i) {
    CloudPoint tmp = p3d_[matches_[id][i]];
    if (tmp.pt.x != 0 || tmp.pt.y != 0 || tmp.pt.z != 0)
      res.push_back (tmp);
  }
  return res;
}

std::vector<DMatch> Image::matches (Image img) {
  vector<DMatch> res;
  for (size_t i =0; i<matches_[img.id_].size(); ++i)
    res.push_back (DMatch (matches_[img.id_][i],
                           img.matches_[id_][i],
                           1));
  return res;
}

void Image::set_image (cv::Mat image) {
  if (!image.data)
    cerr << "Error : Image.set_image : Trying to set void image" << endl;
  image_ = image;
}

void Image::add_keypoints (vector<KeyPoint> keypoints) {
  int n = keypoints.size();
  kpts_.resize (n);
  desc_.resize (n);
  p3d_.resize (n);
  // Voir si besoin d'une boucle for pour tout transferer
  kpts_ = keypoints;
}

void Image::add_descriptors (Mat descriptors) {
  //desc_.resize (kpts_.size());
  desc_ = descriptors;
}

void Image::set_point3d (vector<Point3f> points3d) {
  for (size_t i=0; i<points3d.size(); ++i) {
    CloudPoint cpt;
    cpt.pt = points3d[i];
    p3d_.push_back (cpt);
  }
}

void Image::set_point3d (vector<CloudPoint> points3d) {
  //p3d_.resize (kpts_.size());
  p3d_ = points3d;
}

//void Image::add_point3d (int id, vector<Point3f> points3d) {
void Image::add_point3d (int id, vector<CloudPoint> points3d) {
  for (size_t i=0; i<matches_[id].size(); ++i) {
    //cout << points3d[i] << endl;
    p3d_[matches_[id][i]] = points3d[i];
  }
}

void Image::add_match (int image_id, std::vector<unsigned int> index) {
  matches_.insert (std::pair< int, vector<unsigned int> > (image_id, index));
}

void Image::set_pose (Matx34d P) {
  P_ = P;
}

cv::Mat Image::draw_matches (Image img) {
  Mat out;

  vector<DMatch> match = matches(img);

  drawMatches (image_, kpts_, img.image_, img.kpts_, match, out);
  imshow ("matches", out);
  waitKey(0);

  return out;
}
