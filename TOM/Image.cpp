#include "Image.h"

using namespace std;
using namespace cv;

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

vector<Point3f> Image::p3d (int id) {
  vector<Point3f> res;
  for (int i=0; i<matches_[id].size(); ++i) {
    Point3f tmp = p3d_[matches_[id][i]];
    if (tmp.x != 0 || tmp.y != 0 || tmp.z != 0)
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
  kpts_.resize (keypoints.size());
  // Voir si besoin d'une boucle for pour tout transferer
  kpts_ = keypoints;
}

void Image::add_descriptors (Mat descriptors) {
  desc_.resize (kpts_.size());
  desc_ = descriptors;
}

void Image::add_point3d (vector<Point3f> points3d) {
  p3d_.resize (kpts_.size());
  p3d_ = points3d;
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
