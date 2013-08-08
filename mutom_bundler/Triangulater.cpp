#include "Triangulater.h"

using namespace std;
using namespace cv;

int Triangulater::triangulate_bundler (vector<KeyPoint> kpts1, vector<int> &idx1,
                                         vector<KeyPoint> kpts2, vector<int> &idx2,
                                         Matx33d cvR0, Matx31d cvt0,
                                         Matx33d cvR1, Matx31d cvt1,
                                         Mat Kin, vector<Point3d>& pointcloud)
{
  pointcloud.clear();

  int n = idx1.size();
  int n_pos = 0;

  bool in_front = true;
  double mean_proj_error = 0.0;
  double proj_error = 0.0;
  double angle = 0.0;

  v2_t* vp1 = (v2_t*) malloc (n*sizeof(v2_t));
  v2_t* vp2 = (v2_t*) malloc (n*sizeof(v2_t));
  v3_t* vp3d = (v3_t*) malloc (n*sizeof (v3_t));
  double *R0 = (double*) malloc (9*sizeof(double));
  double *t0 = (double*) malloc (9*sizeof(double));
  double *R = (double*) malloc (9*sizeof(double));
  double *t = (double*) malloc (9*sizeof(double));
  double *K = (double*) malloc (9*sizeof(double));

  cv2bd( idx2pts(kpts1, idx1), vp1);
  cv2bd( idx2pts(kpts2, idx2), vp2);
  mat_cv2bd(cvR0, R0);
  mat_cv2bd(cvt0, t0);
  mat_cv2bd(cvR1, R);
  mat_cv2bd(cvt1, t);
  mat_cv2bd(Kin, K);

  camera_params_t c0 = camera_params_new(R0, t0, K);
  camera_params_t c1 = camera_params_new(R, t, K);

  for (int i=0; i<n; ++i)
  {
    proj_error = 0;
    angle = 0;
    // Triangulate two points
    vp3d[i] = triangulate_robust (vp1[i], vp2[i],
                                  c0,  c1,
                                  proj_error, in_front, angle,
                                  true);

    pointcloud.push_back (Point3d (Vx (vp3d[i]), Vy (vp3d[i]), Vz (vp3d[i])));
    //cout << "Err : " << proj_error << " /In front :" << in_front << " /Angle : " << angle << endl;
    if( Vz (vp3d[i]) > 0 )
    {
      ++n_pos;
    }

    mean_proj_error += proj_error;
//    else
//    {
//      idx1.erase(idx1.begin()+i);
//      idx2.erase(idx2.begin()+i);
//    }

  }

  cout << "Mean projection error :" << mean_proj_error/n << endl;

  return n_pos;
}

int Triangulater::triangulate_simple (vector<KeyPoint> kpts1, vector<int> idx1,
                                       vector<KeyPoint> kpts2, vector<int> idx2,
                                       Matx33d R0, Matx31d t0,
                                       Matx33d R1, Matx31d t1,
                                       Mat Kinv, vector<Point3d>& pointcloud)
{
	pointcloud.clear();
  vector<Point2d> pts1 = idx2pts(kpts1, idx1);
  vector<Point2d> pts2 = idx2pts(kpts2, idx2);

	size_t pts_size = pts1.size();
	for (size_t i=0; i<pts_size; ++i)
	{
		Point2f kp = pts1[i];
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kinv * Mat_<double>(u);
		u = um.at<Point3d>(0);
		Point2f kp1 = pts2[i];
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		u1 = um1.at<Point3d>(0);

    Matx34d P = Matx34d(R0(0,0), R0(0,1), R0(0,2), t0(0,0),
                        R0(1,0), R0(1,1), R0(1,2), t0(1,0),
                        R0(2,0), R0(2,1), R0(2,2), t0(2,0));
    Matx34d P1 = Matx34d(R1(0,0), R1(0,1), R1(0,2), t1(0,0),
                        R1(1,0), R1(1,1), R1(1,2), t1(1,0),
                        R1(2,0), R1(2,1), R1(2,2), t1(2,0));
		Mat_<double> X = iterativeLinearLSTriangulation(u,P,u1,P1);

    pointcloud.push_back (Point3d (X (0), X (1), X (2)));
	}

	return pointcloud.size ();
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> Triangulater::linearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
                                                   Matx34d P,		//camera 1 matrix
                                                   Point3d u1,		//homogenous image point in 2nd camera
                                                   Matx34d P1		//camera 2 matrix
                                                   )
{
	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	Mat_<double> B = (Mat_<double>(4,1) <<	-(u.x*P(2,3)	-P(0,3)),
					  -(u.y*P(2,3)	-P(1,3)),
					  -(u1.x*P1(2,3)	-P1(0,3)),
					  -(u1.y*P1(2,3)	-P1(1,3)));

	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);

	return X;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> Triangulater::iterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
                                                            Matx34d P,			//camera 1 matrix
                                                            Point3d u1,			//homogeneous image point in 2nd camera
                                                            Matx34d P1			//camera 2 matrix
                                                            ) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4,1);
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = linearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	-(u.x*P(2,3)	-P(0,3))/wi,
						  -(u.y*P(2,3)	-P(1,3))/wi,
						  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
						  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );

		solve(A,B,X_,DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}



/* Triangulate two points */
v3_t Triangulater::triangulate_robust(v2_t p, v2_t q,
                                      camera_params_t c1, camera_params_t c2,
                                      double &proj_error, bool &in_front, double &angle,
                                      bool explicit_camera_centers)
{
  double K1[9], K2[9];
  double K1inv[9], K2inv[9];

  GetIntrinsics(c1, K1);
  GetIntrinsics(c2, K2);

  matrix_invert(3, K1, K1inv);
  matrix_invert(3, K2, K2inv);

  /* Set up the 3D point */
  double proj1[3] = { Vx(p), Vy(p), 1.0 };
  double proj2[3] = { Vx(q), Vy(q), 1.0 };

  double proj1_norm[3], proj2_norm[3];

  matrix_product(3, 3, 3, 1, K1inv, proj1, proj1_norm);
  matrix_product(3, 3, 3, 1, K2inv, proj2, proj2_norm);

  v2_t p_norm = v2_new(proj1_norm[0] / proj1_norm[2],
     proj1_norm[1] / proj1_norm[2]);

  v2_t q_norm = v2_new(proj2_norm[0] / proj2_norm[2],
     proj2_norm[1] / proj2_norm[2]);

  /* Compute the angle between the rays */
  angle = ComputeRayAngle(p, q, c1, c2);

  /* Triangulate the point */
  v3_t pt;
  if (!explicit_camera_centers)
  {
    pt = triangulate(p_norm, q_norm, c1.R, c1.t, c2.R, c2.t, &proj_error);
  }
  else
  {
    double t1[3];
    double t2[3];

    /* Put the translation in standard form */
    matrix_product(3, 3, 3, 1, c1.R, c1.t, t1);
    matrix_scale(3, 1, t1, -1.0, t1);
    matrix_product(3, 3, 3, 1, c2.R, c2.t, t2);
    matrix_scale(3, 1, t2, -1.0, t2);

    pt = triangulate(p_norm, q_norm, c1.R, t1, c2.R, t2, &proj_error);
  }

  proj_error = (c1.f + c2.f) * 0.5 * sqrt(proj_error * 0.5);

  /* Check cheirality */
  bool cc1 = CheckCheirality(pt, c1);
  bool cc2 = CheckCheirality(pt, c2);

  in_front = (cc1 && cc2);

  return pt;
}

void Triangulater::GetIntrinsics (const camera_params_t &camera, double *K)
{
    if (!camera.known_intrinsics) {
        K[0] = camera.f;  K[1] = 0.0;       K[2] = 0.0;
        K[3] = 0.0;       K[4] = camera.f;  K[5] = 0.0;
        K[6] = 0.0;       K[7] = 0.0;       K[8] = 1.0;
    } else {
        memcpy(K, camera.K_known, 9 * sizeof(double));
    }
}

/* Compute the angle between two rays */
double Triangulater::ComputeRayAngle(v2_t p, v2_t q,
                                       const camera_params_t &cam1,
                                       const camera_params_t &cam2)
{
    double K1[9], K2[9];
    GetIntrinsics(cam1, K1);
    GetIntrinsics(cam2, K2);

    double K1_inv[9], K2_inv[9];
    matrix_invert(3, K1, K1_inv);
    matrix_invert(3, K2, K2_inv);

    // EDIT!!! Changed -1 to 1
    double p3[3] = { Vx(p), Vy(p), 1.0 };
    double q3[3] = { Vx(q), Vy(q), 1.0 };

    double p3_norm[3], q3_norm[3];
    matrix_product331(K1_inv, p3, p3_norm);
    matrix_product331(K2_inv, q3, q3_norm);

    v2_t p_norm = v2_new(p3_norm[0] / p3_norm[2], p3_norm[1] / p3_norm[2]);
    v2_t q_norm = v2_new(q3_norm[0] / q3_norm[2], q3_norm[1] / q3_norm[2]);

    double R1_inv[9], R2_inv[9];
    matrix_transpose(3, 3, (double *) cam1.R, R1_inv);
    matrix_transpose(3, 3, (double *) cam2.R, R2_inv);

    double p_w[3], q_w[3];

    // EDIT!!! Changed -1 to 1
    double pv[3] = { Vx(p_norm), Vy(p_norm), 1.0 };
    double qv[3] = { Vx(q_norm), Vy(q_norm), 1.0 };

    double Rpv[3], Rqv[3];

    matrix_product331(R1_inv, pv, Rpv);
    matrix_product331(R2_inv, qv, Rqv);

    matrix_sum(3, 1, 3, 1, Rpv, (double *) cam1.t, p_w);
    matrix_sum(3, 1, 3, 1, Rqv, (double *) cam2.t, q_w);

    /* Subtract out the camera center */
    double p_vec[3], q_vec[3];
    matrix_diff(3, 1, 3, 1, p_w, (double *) cam1.t, p_vec);
    matrix_diff(3, 1, 3, 1, q_w, (double *) cam2.t, q_vec);

    /* Compute the angle between the rays */
    double dot;
    matrix_product(1, 3, 3, 1, p_vec, q_vec, &dot);

    double mag = matrix_norm(3, 1, p_vec) * matrix_norm(3, 1, q_vec);

    return acos(CLAMP(dot / mag, -1.0 + 1.0e-8, 1.0 - 1.0e-8));
}

/* Check cheirality for a camera and a point */
bool Triangulater::CheckCheirality(v3_t p, const camera_params_t &camera)
{
    double pt[3] = { Vx(p), Vy(p), Vz(p) };
    double cam[3];

    pt[0] -= camera.t[0];
    pt[1] -= camera.t[1];
    pt[2] -= camera.t[2];
    matrix_product(3, 3, 3, 1, (double *) camera.R, pt, cam);

    // EDIT!!!
    if (cam[2] > 0.0)
        return false;
    else
        return true;
}


Triangulater::Triangulater ()
{}

Triangulater::~Triangulater ()
{}
