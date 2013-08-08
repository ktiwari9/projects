#include "triangulation.h"

/* Triangulate two points */
v3_t triangulate_bundler(v2_t p, v2_t q,
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
  // EDIT!!!
  double proj1[3] = { Vx(p), Vy(p), -1.0 };
  double proj2[3] = { Vx(q), Vy(q), -1.0 };

  double proj1_norm[3], proj2_norm[3];

  matrix_product(3, 3, 3, 1, K1inv, proj1, proj1_norm);
  matrix_product(3, 3, 3, 1, K2inv, proj2, proj2_norm);

  v2_t p_norm = v2_new(proj1_norm[0] / proj1_norm[2],
     proj1_norm[1] / proj1_norm[2]);

  v2_t q_norm = v2_new(proj2_norm[0] / proj2_norm[2],
     proj2_norm[1] / proj2_norm[2]);

  /* Undo radial distortion */
  //p_norm = UndistortNormalizedPoint(p_norm, c1);
  //q_norm = UndistortNormalizedPoint(q_norm, c2);

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

void GetIntrinsics (const camera_params_t &camera, double *K)
{
    if (!camera.known_intrinsics) {
        K[0] = camera.f;  K[1] = 0.0;       K[2] = 0.0;
        K[3] = 0.0;       K[4] = camera.f;  K[5] = 0.0;
        K[6] = 0.0;       K[7] = 0.0;       K[8] = 1.0;
    } else {
        memcpy(K, camera.K_known, 9 * sizeof(double));
    }
}

v2_t UndistortNormalizedPoint(v2_t p, camera_params_t c)
{
    double r = sqrt(Vx(p) * Vx(p) + Vy(p) * Vy(p));
    if (r == 0.0)
        return p;

    double t = 1.0;
    double a = 0.0;

    for (int i = 0; i < POLY_INVERSE_DEGREE; i++) {
        a += t * c.k_inv[i];
        t = t * r;
    }

    double factor = a / r;

    return v2_scale(factor, p);
}

/* Compute the angle between two rays */
double ComputeRayAngle(v2_t p, v2_t q,
                       const camera_params_t &cam1,
                       const camera_params_t &cam2)
{
    double K1[9], K2[9];
    GetIntrinsics(cam1, K1);
    GetIntrinsics(cam2, K2);

    double K1_inv[9], K2_inv[9];
    matrix_invert(3, K1, K1_inv);
    matrix_invert(3, K2, K2_inv);

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

    double pv[3] = { Vx(p_norm), Vy(p_norm), -1.0 };
    double qv[3] = { Vx(q_norm), Vy(q_norm), -1.0 };

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
bool CheckCheirality(v3_t p, const camera_params_t &camera)
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
