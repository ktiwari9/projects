#pragma once

#include "SynPNP.h"
#include "Pipeline.h"

const int n = 100;
const double noise = 5;

const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;
const double a = 0;
const double b = 0;
const double c = 0;
const double d = 0;
const double e = 0;

void test_matches ();

void setup_test_data ();
void test_pose2D2D ();
void test_pose3D2D ();
void test_triangulation ();
void test_bundle ();
void test_sfm_two_images ();
