#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

vector<Point2f> corners;

void mouse_cb( int event, int x, int y, int flags, void* param )
{
  // If button released
  if (event == CV_EVENT_LBUTTONUP)
  {
    if (corners.size () >= 4)
      corners.clear ();

    corners.push_back ( Point2f(x, y));
    cout << "Corner " << corners.size () << " saved." << endl;
  }
}

void rect_img (Mat in, Mat &out, vector<Point2f> corners, float width, float height)
{
  // Corners in rectified image
  vector<Point2f> dest;
  dest.push_back (Point2f (0.0f, 0.0f));
  dest.push_back (Point2f (width, 0.0f));
  dest.push_back (Point2f (0.0f, height));
  dest.push_back (Point2f (width, height));

  Mat T = getPerspectiveTransform (corners, dest);
  Mat warped;
  warpPerspective (in, out, T, Size (width, height));
}

int main()
{
  float width = 93; // mm
  float height = 244; // mm
  char* image = "/home/gmanfred/Pictures/kangos.jpg";
  Mat img = imread(image);
  if (img.empty ())
  {
    cout << "Can't find/open image." << endl;
    return -1;
  }

  namedWindow ("Input");
  setMouseCallback ( "Input", mouse_cb, (void*) image);

  imshow ("Input", img);
  waitKey(0);

  // Create destination points
  Mat warped;
  rect_img (img, warped, corners, width, height);

  namedWindow ("Output");
  imshow("Output", warped);
  waitKey(0);

  destroyAllWindows();
  return 0;
}
