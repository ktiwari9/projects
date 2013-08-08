/* 
 *  Copyright (c) 2008  Noah Snavely (snavely (at) cs.washington.edu)
 *    and the University of Washington
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 */

/* 5point.h */
/* Solve the 5-point relative pose problem */

#ifndef __5POINT_H__
#define __5POINT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vector.h"

int compute_pose_ransac(int n, v2_t *r_pts, v2_t *l_pts, 
                        double *K1, double *K2, 
                        double ransac_threshold, int ransac_rounds, 
                        double *R_out, double *t_out, double* inlier_pts);


void generate_Ematrix_hypotheses(int n, v2_t *rt_pts, v2_t *left_pts, 
                                 int *num_poses, double *E);
                                 
void choose(int n, int k, int *arr);

int evaluate_Ematrix(int n, v2_t *r_pts, v2_t *l_pts, double thresh_norm,
                     double *F, int *best_inlier, double *score, double* inliers);                                 

#ifdef __cplusplus
}
#endif

#endif /* __5POINT_H__ */
