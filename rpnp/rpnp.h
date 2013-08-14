/**
Copyright (c) 2013, LAAS-CNRS
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the LAAS-CNRS the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE LAAS-CNRS AND CONTRIBUTORS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE LAAS-CNRS AND CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Guido Manfredi, LAAS-CNRS, August 2013
*/
/// author : Guido Manfredi
/// mail : gmanfredi.mail@gmail.com

#ifndef RPNP_RPNP_H_
#define RPNP_RPNP_H_
// included just for the SVD function
#include <opencv2/core/core.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
/**
 * @brief The rpnp class implements the algorithms described in
 *  A Robust O(n) Solution to the Perspective-n-Point Problem,
 *  Shiqi Li, Chi Xu, and Ming Xie, Member, IEEE
 * It allows setting intrinsic parameters, 2D/3D points correspondences
 * and compute the corresponding pose.
 * This class also offers functions to get the relative error between
 * two camera poses or compute the reprojection error for a given pose.
 * This API is strongly based on the implementation of epnp,
 * by Vincent Lepetit.
**/
class rpnp {
 public:
  rpnp(void);
  ~rpnp();
  /**
  * @brief Use this function to set the intrinsic parameters of the
  *       camera.
  **/
  void set_internal_parameters(const double uc, const double vc,
			       const double fu, const double fv);
  /**
   * @brief Before adding correspondences, you need to set a maximum
   *       number of correspondences.
  **/
  void set_maximum_number_of_correspondences(const int n);
  /**
   * @brief Clears the currespondences inside the class.
  **/
  void reset_correspondences(void);
  /**
   * @brief Takes a 3D point coordinates (X, Y, Z) and a 2D point
   *        coordinates (u, v) and save them as a correspondence.
  **/
  void add_correspondence(const double X, const double Y, const double Z,
			  const double u, const double v);
  /**
   * @brief Uses the given correspondences to compute
   *        a rotation matrix R and a translation matrix T.
  **/
  double compute_pose(double R[3][3], double T[3]);
  /**
   * @brief Given two rotation and translation matrices, compute the
   *        relative translation and rotation error.
  **/
  void relative_error(const double Rtrue[3][3], const double ttrue[3],
                      const double Rest[3][3],  const double test[3],
                      double & rot_err, double & transl_err);
  /**
   * @brief Print to stdout the given matrices.
  **/
  void print_pose(const double R[3][3], const double t[3]);
  /**
   * @brief Computes the reprojection error
  **/
  double reprojection_error(double R[3][3], double t[3]);

 private:
  void select_longest_edge ();
  void compute_rotation_matrix ();
  void transform_points_to_new_frame ();
  void divide_into_3points_set ();
  void retrieve_local_minima ();
  double compute_camera_poses_minima (double R[3][3], double t[3]);

  void getp3p (double l1, double l2, double A5, double C1, double C2,
                double D1, double D2, double D3,
                double D4[5]);
  void getpoly7 (double F[5], double F7[8]);
  void calcampose (double * XXc, double * XXw,
                   double R[3][3], double t[3]);
  void polyval (double D6[7], double t2s[7], double F[7]);
  double norm (double u[3]);
  double norm (double u, double v, double w);
  void xcross(double a[3], double b[3],
              double c[3]);
  void roots (double D7[8],
              double real[7], double imag[7]);
  void eig (double DTD[6][6], double V[6][6], double D[6][6]);
  double max_array (double t2s[7]);
  void mat_to_quat(const double R[3][3], double q[4]);
  double dot(const double * v1, const double * v2);
  void log (std::string msg);

  double uc_, vc_, fu_, fv_;
  double * XX_, *XXc_, * XXa_, * xx_, * xxv_;
  int maximum_number_of_correspondences_;
  int number_of_correspondences_;
  int i1_, i2_;
  double P0_[3];
  double R_[3][3];
  double D7_[7];
  double t2s_[7];
};

#endif
