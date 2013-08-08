#include "test_pipeline.h"

using namespace cv;
using namespace std;

SynPNP sim (uc, vc, fu, fv);
Pipeline pl (uc, vc, fu, fv, a, b, c, d, e);

void test_matches () {
  string path_im1 = "/home/gmanfred/Pictures/box_in_scene.png";
  string path_im2 = "/home/gmanfred/Pictures/box.png";

  Mat im1, im2;
  im1 = imread (path_im1, CV_LOAD_IMAGE_GRAYSCALE);
  im2 = imread (path_im2, CV_LOAD_IMAGE_GRAYSCALE);
  Image image1(im1, 0), image2(im2, 1);
  pl.extract (image1);
  pl.extract (image2);
  pl.match (image1, image2);
  image1.draw_matches(image2);
}

void test_pose2D2D () {
  // CREATING IMAGES
  Mat im1, im2;
  string path_im1 = "/home/gmanfred/Pictures/box_in_scene.png";
  string path_im2 = "/home/gmanfred/Pictures/box.png";
  im1 = imread (path_im1, CV_LOAD_IMAGE_GRAYSCALE);
  im2 = imread (path_im2, CV_LOAD_IMAGE_GRAYSCALE);
  Image image1(im1, 0), image2(im2, 1);

	// SIMULATING MATCHES
	vector<Point3f> p3d;
	for (unsigned int i = 0; i < n; ++i) {
		double X, Y, Z;
		sim.random_point (X, Y, Z);
		p3d.push_back (Point3d(X, Y, Z));
	}
	// POSES
	double R1[3][3], R2[3][3], t1[3], t2[3];
	sim.identity (R1, t1);
	sim.random_pose (R2, t2);
	// P2D
	vector<Point2f> p2d1, p2d2;
	for (unsigned int i = 0; i < n; ++i) {
		double x1, y1, x2, y2;
		sim.project_with_noise (R1, t1, noise, p3d[i].x, p3d[i].y, p3d[i].z, x1, y1);
		sim.project_with_noise (R2, t2, noise, p3d[i].x, p3d[i].y, p3d[i].z, x2, y2);
		p2d1.push_back (Point2d(x1, y1));
		p2d2.push_back (Point2d(x2, y2));
	}
	std::vector<cv::KeyPoint> kpts1, kpts2;
	PointsToKeyPoints(p2d1, kpts1);
	PointsToKeyPoints(p2d2, kpts2);
	image1.add_keypoints (kpts1);
	image2.add_keypoints (kpts2);
	std::vector<unsigned int> matches1 (kpts1.size());
	std::vector<unsigned int> matches2 (kpts1.size());
	for (size_t i=0; i<kpts1.size(); ++i) {
    matches1[i] = i;
    matches2[i] = i;
	}
  image1.add_match (image2.id_, matches1);
  image2.add_match (image1.id_, matches2);
  // COMPUTING
	Matx34d P;
	pl.find_camera_matrix2D2D(image1, image2, P);

	cout << R2[0][0] << " " << R2[0][1] << " " << R2[0][2] << endl
			 << R2[1][0] << " " << R2[1][1] << " " << R2[1][2] << endl
			 << R2[2][0] << " " << R2[2][1] << " " << R2[2][2] << endl;
	cout << t2[0] << " " << t2[1] << " " << t2[2] << endl;

	cout << P << endl;
}

void test_pose3D2D () {
  // CREATING IMAGES
  Mat im1, im2;
  string path_im1 = "/home/gmanfred/Pictures/box_in_scene.png";
  string path_im2 = "/home/gmanfred/Pictures/box.png";
  im1 = imread (path_im1, CV_LOAD_IMAGE_GRAYSCALE);
  im2 = imread (path_im2, CV_LOAD_IMAGE_GRAYSCALE);
  Image image1(im1, 0), image2(im2, 1);

	// SIMULATING MATCHES
	vector<Point3f> p3d;
	for (unsigned int i = 0; i < n; ++i) {
		double X, Y, Z;
		sim.random_point (X, Y, Z);
		p3d.push_back (Point3d(X, Y, Z));
	}
	// POSES
	double R1[3][3], R2[3][3], t1[3], t2[3];
	sim.identity (R1, t1);
	sim.random_pose (R2, t2);
	// P2D
	vector<Point2f> p2d1, p2d2;
	for (unsigned int i = 0; i < n; ++i) {
		double x1, y1, x2, y2;
		sim.project_with_noise (R1, t1, noise, p3d[i].x, p3d[i].y, p3d[i].z, x1, y1);
		sim.project_with_noise (R2, t2, noise, p3d[i].x, p3d[i].y, p3d[i].z, x2, y2);
		p2d1.push_back (Point2d(x1, y1));
		p2d2.push_back (Point2d(x2, y2));
	}
	std::vector<cv::KeyPoint> kpts1, kpts2;
	PointsToKeyPoints(p2d1, kpts1);
	PointsToKeyPoints(p2d2, kpts2);
	image1.add_keypoints (kpts1);
	image2.add_keypoints (kpts2);
	std::vector<unsigned int> matches1 (kpts1.size());
	std::vector<unsigned int> matches2 (kpts1.size());
	for (size_t i=0; i<kpts1.size(); ++i) {
    matches1[i] = i;
    matches2[i] = i;
	}
  image1.add_match (image2.id_, matches1);
  image1.add_point3d (p3d);
  image2.add_match (image1.id_, matches2);
  // COMPUTING
	Matx34d P;
	pl.find_camera_matrix3D2D(image1, image2, P);

	cout << R2[0][0] << " " << R2[0][1] << " " << R2[0][2] << endl
			 << R2[1][0] << " " << R2[1][1] << " " << R2[1][2] << endl
			 << R2[2][0] << " " << R2[2][1] << " " << R2[2][2] << endl;
	cout << t2[0] << " " << t2[1] << " " << t2[2] << endl;

	cout << P << endl;
}
