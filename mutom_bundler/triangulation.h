#ifndef DEF_MUTOM_BUNDLER_TRIANGULATION
#define DEF_MUTOM_BUNDLER_TRIANGULATION

#include <cmath>
#include <cstring>

#include "matrix/matrix.h"
#include "matrix/vector.h"
#include "sfm-driver/sfm.h"
#include "imagelib/triangulate.h"
#include "imagelib/defines.h"

/* Triangulate two points */
v3_t triangulate_bundler(v2_t p, v2_t q,
                          camera_params_t c1, camera_params_t c2,
                          double &proj_error, bool &in_front, double &angle,
                          bool explicit_camera_centers);

void GetIntrinsics (const camera_params_t &camera, double *K);
v2_t UndistortNormalizedPoint(v2_t p, camera_params_t c);
/* Compute the angle between two rays */
double ComputeRayAngle(v2_t p, v2_t q,
                       const camera_params_t &cam1,
                       const camera_params_t &cam2);

bool CheckCheirality(v3_t p, const camera_params_t &camera);

#endif // DEF_MUTOM_BUNDLER_TRIANGULATION
