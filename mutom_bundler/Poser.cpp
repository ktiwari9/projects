#include <vector>
#include <opencv2/calib3d/calib3d.hpp>

#include "Poser.h"

using namespace std;
using namespace cv;

int Poser::estimate_pose_5point (vector<KeyPoint> kpts1, vector<int> &idx1,
                                   vector<KeyPoint> kpts2, vector<int> &idx2,
                                   double scale, Mat Kin,
                                   Matx33d &R_out, Matx31d &t_out)
{
	int n = idx1.size ();
	int n_in = 0;
	v2_t* vp1 = (v2_t*) malloc (n*sizeof (v2_t));
	v2_t* vp2 = (v2_t*) malloc (n*sizeof (v2_t));
	double* inliers = (double*) malloc (n*sizeof(double));
	double* K = (double*) malloc (9*sizeof(double));
	double* R = (double*) malloc (9*sizeof(double));
	double* t = (double*) malloc (3*sizeof(double));

  cv2bd ( idx2pts (kpts1, idx1), vp1 );
  cv2bd ( idx2pts (kpts2, idx2), vp2 );
  mat_cv2bd(Kin, K);

  n_in = compute_pose_ransac(n, vp1, vp2, K, K, m_ransac_thresh, m_ransac_rounds, R, t, inliers);

  mat_bd2cv(R, R_out);
  t_out (0,0) = scale*t[0]; t_out (1,0) = scale*t[1]; t_out (2,0) = scale*t[2];

  // Filtering outliers
  vector<int> good_idx1;
  vector<int> good_idx2;
  for(int i=0; i < n_in; ++i)
	{
	  good_idx1.push_back(idx1[inliers[i]]);
	  good_idx2.push_back(idx2[inliers[i]]);
	}

  idx1 = good_idx1;
  idx2 = good_idx2;

	return n_in;
}

int Poser::estimate_pose_3point_cv (vector<Point3d> p3d, vector<int> idx_3d,
                                 vector<KeyPoint> kpts, vector<int> idx_2d,
                                 cv::Mat Kin, Matx33d &R, Matx31d &t)
{
	vector<Point3f> pts_3d = d2f (idx2pts (p3d, idx_3d));
	vector<Point2f> pts_2d = d2f (idx2pts (kpts, idx_2d));

	Matx31d Rvec, tvec;

	vector<int> inliers;
//	for(size_t i=0; i<pts_3d.size(); ++i)
//    cout << pts_3d[i].x << " " << pts_3d[i].y << " " << pts_3d[i].z << endl;
	solvePnPRansac (Mat(pts_3d), Mat(pts_2d), Kin, Mat(), Rvec, tvec, false,
                  1024, 4.0, 100, inliers, CV_ITERATIVE);
	//solvePnP (Mat(pts_3d), Mat(pts_2d), Kin, Mat(), Rvec, tvec, CV_EPNP);

	Rodrigues (Rvec, R);

  t = Matx31d ( tvec(0,0),
                tvec(1,0),
                tvec(2,0));

  return inliers.size ();
}

int Poser::estimate_pose_3point_bd (vector<Point3d> p3d, vector<int> idx_3d,
                                 vector<KeyPoint> kpts, vector<int> idx_2d,
                                 cv::Mat Kin, Matx33d &R, Matx31d &t)
{
  int n = idx_3d.size ();
  v3_t *points = (v3_t*) malloc (n*sizeof (v3_t));
  v2_t *projs = (v2_t*) malloc (n*sizeof (v2_t));
  cv2bd ( idx2pts (p3d, idx_3d), points);
  cv2bd ( idx2pts (kpts, idx_2d), projs);

  double *P = (double*) malloc (12*sizeof (double));
  find_projection_3x4_ransac(n, points, projs, P, 2048, 4.0);

  R = Matx33d (P[0], P[1], P[2],
               P[4], P[5], P[6],
               P[8], P[9], P[10]);

  t = Matx31d (P[3], P[7], P[11]);
}

Poser::Poser ()
{
	m_ransac_inliers = 0.99;
	m_ransac_rounds = 1024;
	m_ransac_thresh = 2.0;
}

Poser::~Poser ()
{}
