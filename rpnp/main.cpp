// author : guido manfred
// mail : gmanfredi.mail@gmail.com



#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;

#include "rpnp.h"

const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;

const int n = 100;
const double noise = 10;

double rand(double min, double max)
{
  return min + (max - min) * double(rand()) / RAND_MAX;
}

void random_pose(double R[3][3], double t[3])
{
  const double range = 1;

  double phi   = rand(0, range * 3.14159 * 2);
  double theta = rand(0, range * 3.14159);
  double psi   = rand(0, range * 3.14159 * 2);

  R[0][0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
  R[0][1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
  R[0][2] = sin(psi) * sin(theta);

  R[1][0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
  R[1][1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
  R[1][2] = cos(psi) * sin(theta);

  R[2][0] = sin(theta) * sin(phi);
  R[2][1] = -sin(theta) * cos(phi);
  R[2][2] = cos(theta);

  t[0] = 7.0f;
  t[1] = -50.0f;
  t[2] = 2000.0f;
}

void random_point(double & Xw, double & Yw, double & Zw)
{
  double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159), R = rand(0, +2);

  Xw = sin(theta) * sin(phi) * R;
  Yw = -sin(theta) * cos(phi) * R;
  Zw =  cos(theta) * R;
}

void project_with_noise(double R[3][3], double t[3],
			double Xw, double Yw, double Zw,
			double & u, double & v)
{
  double Xc = R[0][0] * Xw + R[0][1] * Yw + R[0][2] * Zw + t[0];
  double Yc = R[1][0] * Xw + R[1][1] * Yw + R[1][2] * Zw + t[1];
  double Zc = R[2][0] * Xw + R[2][1] * Yw + R[2][2] * Zw + t[2];

  double nu = rand(-noise, +noise);
  double nv = rand(-noise, +noise);
  u = uc + fu * Xc / Zc + nu;
  v = vc + fv * Yc / Zc + nv;
}

int main(int argc, char ** argv)
{
  rpnp PnP;

  srand(time(0));

  PnP.set_internal_parameters(uc, vc, fu, fv);
  PnP.set_maximum_number_of_correspondences(n);

  double R_true[3][3], t_true[3];
  random_pose(R_true, t_true);

  PnP.reset_correspondences();
  for(int i = 0; i < n; i++) {
    double Xw, Yw, Zw, u, v;

    random_point(Xw, Yw, Zw);

    project_with_noise(R_true, t_true, Xw, Yw, Zw, u, v);
    PnP.add_correspondence(Xw, Yw, Zw, u, v);
  }

  double R_est[3][3], t_est[3];
  double err2 = PnP.compute_pose(R_est, t_est);
  double rot_err, transl_err;

  PnP.relative_error(R_true, t_true, R_est, t_est, rot_err, transl_err);
  cout << ">>> Reprojection error: " << err2 << endl;
  cout << ">>> rot_err: " << rot_err << ", transl_err: " << transl_err << endl;
  cout << endl;
  cout << "'True reprojection error':"
       << PnP.reprojection_error(R_true, t_true) << endl;
  cout << endl;
  cout << "True pose:" << endl;
  PnP.print_pose(R_true, t_true);
  cout << endl;
  cout << "Found pose:" << endl;
  PnP.print_pose(R_est, t_est);

  return 0;
}

/*
void read_file (string filepath);

const double uc = 330.987861185693;
const double vc = 243.99226997026;
const double fu = 740.922948418132;
const double fv = 740.470221775282;

rpnp PNP;

int main()
{
  PNP.set_internal_parameters(uc, vc, fu, fv);
  PNP.set_maximum_number_of_correspondences(484);
  read_file ("/home/gmanfred/devel/these/projects/Carod/debug.txt");
  double R[3][3];
  double t[3];
  cout << PNP.compute_pose(R, t) << endl;
  PNP.print_pose(R, t);

  return 0;
}

void read_file (string filepath)
{
  ifstream file;
  file.open (filepath.c_str(), ifstream::in);
  string line;
  vector<string> coordinates;
  if (file.is_open()) {
    while (getline(file, line)) {
      boost::split (coordinates, line, boost::is_any_of (" "));
      PNP.add_correspondence(atof(coordinates[0].c_str()),
                             atof(coordinates[1].c_str()),
                             atof(coordinates[2].c_str()),
                             atof(coordinates[3].c_str()),
                             atof(coordinates[4].c_str()));
    }
  }
  else
    cerr << "couldn't open file" << endl;
}
*/
