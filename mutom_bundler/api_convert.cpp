#include "api.h"

using namespace std;
using namespace cv;

void cv2bd ( vector<Point2d> in, v2_t* out)
{
  int n = in.size ();
  for (int i=0; i<n; ++i)
    out[i] = v2_new (in[i].x, in[i].y);
}

void cv2bd ( vector<Point3d> in, v3_t* out)
{
  int n = in.size ();
  for (int i=0; i<n; ++i)
    out[i] = v3_new (in[i].x, in[i].y, in[i].z);
}

void bd2cv ( int n, v2_t* in, vector<Point2f> &out)
{
  for (int i=0; i<n; ++i)
    out.push_back( Point2f (Vx (in[i]), Vy (in[i])));
}

void bd2cv ( int n, v3_t* in, vector<Point3f> &out)
{
  for (int i=0; i<n; ++i)
    out.push_back( Point3f (Vx (in[i]), Vy (in[i]), Vz (in[i])));
}

void bd2cv ( int n, v2_t* in, vector<Point2d> &out)
{
  for (int i=0; i<n; ++i)
    out.push_back( Point2d (Vx (in[i]), Vy (in[i])));
}

void bd2cv ( int n, v3_t* in, vector<Point3d> &out)
{
  for (int i=0; i<n; ++i)
    out.push_back( Point3d (Vx (in[i]), Vy (in[i]), Vz (in[i])));
}

void mat_cv2bd (Mat in, double* out)
{
	for (int i=0; i<3;  ++i)
		for (int j=0; j<3; ++j)
			out[i*3+j] = in.at<double>(i,j);
}

void mat_cv2bd (Matx33d in, double* out)
{
	for (int i=0; i<3;  ++i)
		for (int j=0; j<3; ++j)
			out[i*3+j] = in(i,j);
}

void mat_cv2bd (Matx31d in, double* out)
{
	for (int i=0; i<3;  ++i)
			out[i] = in(i,0);
}

void mat_bd2cv (double* in, Matx33d& out)
{
	for (int i=0; i<3;  ++i)
		for (int j=0; j<3; ++j)
			out(i,j) = in[i*3+j];
}

vector<Point2d> idx2pts (vector<KeyPoint> kpts, vector<int> idx)
{
	vector<Point2d> pts;

	for(size_t n = 0; n < idx.size(); ++n)
		pts.push_back(kpts[idx[n]].pt);

	return pts;
}

vector<Point3d> idx2pts (vector<Point3d> p3d, vector<int> idx)
{
	vector<Point3d> pts;

	for(size_t n = 0; n < idx.size(); ++n)
		pts.push_back(p3d[idx[n]]);

	return pts;
}

camera_params_t camera_params_new (double* R, double* t, double* K)
{
  camera_params_t c;
  for (int n=0; n<9; ++n)
    c.R[n] = R[n];
  for (int n=0; n<3; ++n)
    c.t[n] = t[n];
  c.f = K[0];

  c.k[0] = 0.0;
  c.k[1] = 0.0;
  for (int n=0; n<6; ++n)
    c.k_inv[n] = 0.0;

  for (int n=0; n<9; ++n)
    c.constrained[n] = 0.0;
  for (int n=0; n<9; ++n)
    c.constraints[n] = 0.0;
  for (int n=0; n<9; ++n)
    c.weights[n] = 0.0;

  for (int n=0; n<9; ++n)
    c.K_known[n] = K[n];
  for (int n=0; n<5; ++n)
    c.k_known[n] = 0.0;

  c.fisheye = 0;
  c.known_intrinsics = 1;
  c.f_cx = K[0];
  c.f_cy = K[0];
  c.f_rad = 0.0;
  c.f_angle = 0.0;
  c.f_focal = 0.0;

  c.f_scale = 1.0;
  c.k_scale = 1.0;

  return c;
}

vector<Point2f> d2f (vector<Point2d> in)
{
	vector<Point2f> out(in.size());
	for (size_t n=0; n<in.size(); ++n)
	{
		out[n] = Point2f ( in[n].x, in[n].y);
	}

	return out;
}

vector<Point3f> d2f (vector<Point3d> in)
{
	vector<Point3f> out(in.size());
	for (size_t n=0; n<in.size(); ++n)
	{
		out[n] = Point3f ( in[n].x, in[n].y, in[n].z);
	}

	return out;
}
