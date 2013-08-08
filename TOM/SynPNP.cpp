#include "SynPNP.h"

SynPNP::SynPNP (double uc, double vc, double fu, double fv):
uc_(uc), vc_(vc), fu_(fu), fv_(fv) {

}

void SynPNP::random_point(double & Xw, double & Yw, double & Zw)
{
  double theta = randm(0, 3.14159), phi = randm(0, 2 * 3.14159), R = randm(0, +2);

  Xw = sin(theta) * sin(phi) * R;
  Yw = -sin(theta) * cos(phi) * R;
  Zw =  cos(theta) * R;
}

void SynPNP::identity (double R[3][3], double t[3])
{
  R[0][0] = 1;	R[1][0] = 0;  R[2][0] = 0;
  R[0][1] = 0;	R[1][1] = 1;	R[2][1] = 0;
  R[0][2] = 0;	R[1][2] = 0;	R[2][2] = 1;

  t[0] = 0.0f;  t[1] = 0.0f;  t[2] = 0.0f;
}

void SynPNP::random_pose(double R[3][3], double t[3])
{
  const double range = 1;

  double phi   = randm(0, range * 3.14159 * 2);
  double theta = randm(0, range * 3.14159);
  double psi   = randm(0, range * 3.14159 * 2);

  R[0][0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
  R[0][1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
  R[0][2] = sin(psi) * sin(theta);

  R[1][0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
  R[1][1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
  R[1][2] = cos(psi) * sin(theta);

  R[2][0] = sin(theta) * sin(phi);
  R[2][1] = -sin(theta) * cos(phi);
  R[2][2] = cos(theta);

  //t[0] = 0.0f;  t[1] = 0.0f;  t[2] = randm(0.0f, 6.0f);
  t[0] = 0.0f;  t[1] = 0.0f;  t[2] = 6.0f;
}

void SynPNP::project_with_noise(double R[3][3], double t[3], double noise,
								    						double Xw, double Yw, double Zw,
														    double & u, double & v) {
  double Xc = R[0][0] * Xw + R[0][1] * Yw + R[0][2] * Zw + t[0];
  double Yc = R[1][0] * Xw + R[1][1] * Yw + R[1][2] * Zw + t[1];
  double Zc = R[2][0] * Xw + R[2][1] * Yw + R[2][2] * Zw + t[2];

  double nu = randm(-noise, +noise);
  double nv = randm(-noise, +noise);
  u = uc_ + fu_ * Xc / Zc + nu;
  v = vc_ + fv_ * Yc / Zc + nv;
}
void SynPNP::normalize (double &u, double &v) {
	u = (u - uc_)/fu_;
	v = (v - vc_)/fv_;
}
////////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
double SynPNP::randm (double min, double max) {
  return min + (max - min) * double(rand()) / RAND_MAX;
}
