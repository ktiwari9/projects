#ifndef DEF_MUTOM_BUNDLER_TOOLS
#define DEF_MUTOM_BUNDLER_TOOLS

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/point_types.h>

#include "matrix/vector.h"

int matAt(int witdh, int height, int i, int j);
void selectInliers (int n, v2_t* in, double* inliers, v2_t* out);
void printMatx33 (cv::Matx33d M);
void printMatx31 (cv::Matx31d M);
void printMatx33 (double* in);
void print_to_file (char* filename, std::vector<cv::Point3d> p3d);
void print_to_file (char* filename, std::vector<cv::Point2d> pt);
void print_to_file (char* filename, int n_in, v3_t* p3d);
void print_to_file (char* filename, int n_in, v2_t* pt);
void p3d2Cloud (std::vector<cv::Point3d> p3d, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

#endif // DEF_MUTOM_BUNDLER_TOOLS


