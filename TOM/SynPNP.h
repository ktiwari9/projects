#ifndef __TOM_SYNPNP_H_
#define __TOM_SYNPNP_H_

#include <cstdlib>
#include <cmath>

// Synthetique data for PNP testing (or any other computer vision algorithm)
class SynPNP
{
	public:
		// Takes as input the camera center and focal length along u and v.
		SynPNP (double uc, double vc, double fu, double fv);
		// Generate a point at random coordinates in the unit sphere (well, the
		// radius 2 sphere).
		void random_point(double & Xw, double & Yw, double & Zw);
		// Get a identity camera pose, with R = Identity and t = 0.
		void identity (double R[3][3], double t[3]);
		// Get a random camera pose (R t)
		void random_pose(double R[3][3], double t[3]);
		// Project 3D point (X Y Z) on camera at position (R t) and obtain
		// measurements (u v).
		void project_with_noise(double R[3][3], double t[3], double noise,
												    double Xw, double Yw, double Zw,
												    double & u, double & v);
		// Homogeneise (u v) to (u v 1) and multiply it with the inverse of the 
		//intrinsic matrix.
		void normalize (double &u, double &v);

	private:
		// Just a random double generator.
		double randm(double min, double max);
		double uc_, vc_, fu_, fv_;
};

#endif
