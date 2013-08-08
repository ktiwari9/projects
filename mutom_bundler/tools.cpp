#include "tools.h"

using namespace std;

int matAt(int width, int height, int i, int j)
{
  return (i * width + j);
}

void selectInliers (int n, v2_t* in, double* inliers, v2_t* out)
{
  for(int i=0; i<n; ++i)
  {
    //cout << inliers[i] << " ";
    out[i] = in[int(inliers[i])];
  }
  //cout << endl;
}

void printMatx33 (cv::Matx33d M)
{
  for (int i=0; i<3; ++i)
  {
    for (int j=0; j<3; ++j)
    {
      cout << M (i,j) << " ";
    }
    cout << endl;
  }
}

void printMatx31 (cv::Matx31d M)
{
  for (int i=0; i<3; ++i)
  {
      cout << M (i,0) << " ";
  }
  cout << endl;
}

void printMatx33 (double* in)
{
  for (int i=0; i<3; ++i)
  {
    for (int j=0; j<3; ++j)
    {
      cout << in[i*3+j] << " ";
    }
    cout << endl;
  }
}

void print_to_file(char* filename, std::vector<cv::Point3d> p3d)
{
  std::ofstream file;
  file.open (filename);

  for(size_t n=0; n<p3d.size(); ++n)
    file << p3d[n].x << " " << p3d[n].y << " " << p3d[n].z << std::endl;

  file.close();
}

void print_to_file(char* filename, std::vector<cv::Point2d> pt)
{
  std::ofstream file;
  file.open (filename);

  for(size_t n=0; n<pt.size(); ++n)
    file << pt[n].x << " " << pt[n].y << std::endl;

  file.close();
}

void print_to_file(char* filename, int n_in, v3_t* p3d)
{
  std::ofstream file;
  file.open (filename);

  for(size_t n=0; n<n_in; ++n)
    file << Vx(p3d[n]) << " " << Vy(p3d[n]) << " " << Vz(p3d[n]) << std::endl;

  file.close();
}

void print_to_file(char* filename, int n_in, v2_t* pt)
{
  std::ofstream file;
  file.open (filename);

  for(size_t n=0; n<n_in; ++n)
    file << Vx(pt[n]) << " " << Vy(pt[n]) << std::endl;

  file.close();
}

void p3d2Cloud (std::vector<cv::Point3d> p3d, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
	cloud->width  = p3d.size();
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	for (size_t n=0; n<p3d.size(); ++n)
	{
		cloud->points[n].x = p3d[n].x;
		cloud->points[n].y = p3d[n].y;
		cloud->points[n].z = p3d[n].z;
	}
}

