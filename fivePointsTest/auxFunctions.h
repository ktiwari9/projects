#ifndef DEF_AUXFUNCTIONS
#define DEF_AUXFUNCTIONS

#include <iostream>
#include <cstdio>

#include "libmv/multiview/five_point.h"
#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

/******************************************************************
        DISPLAY FUNCTIONS
*******************************************************************/
void print(cv::Mat X);
void print(libmv::Mat X);
void print(libmv::Mat3 X);
void print(libmv::Mat2X X);

#endif // DEF_AUXFUNCTIONS

