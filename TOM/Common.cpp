/*****************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#include "Common.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <assert.h>
#include <sys/types.h>
#include <dirent.h>

#include "Image.h"

using namespace std;
using namespace cv;

std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches) {
	std::vector<cv::DMatch> flip;
	for(int i=0;i<matches.size();i++) {
		flip.push_back(matches[i]);
		swap(flip.back().queryIdx,flip.back().trainIdx);
	}
	return flip;
}

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2)
{
	for (unsigned int i=0; i<matches.size(); i++) {
//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
		assert(matches[i].queryIdx < imgpts1.size());
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		assert(matches[i].trainIdx < imgpts2.size());
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
	kps.clear();
	for (unsigned int i=0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i],1.0f));
}

bool CheckCoherentRotation(cv::Mat_<double>& R) {


	if(fabsf(determinant(R))-1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}

#define intrpmnmx(val,min,max) (max==min ? 0.0 : ((val)-min)/(max-min))

void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, const vector<float>& verror, const Scalar& _line_color)
{
	double minVal,maxVal; minMaxIdx(verror,&minVal,&maxVal,0,0,status);
	int line_thickness = 1;

    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
			double alpha = intrpmnmx(verror[i],minVal,maxVal); alpha = 1.0 - alpha;
			Scalar line_color(alpha*_line_color[0],
							  alpha*_line_color[1],
							  alpha*_line_color[2]);

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

bool hasEndingLower (string const &fullString_, string const &_ending)
{
	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
	return hasEnding(fullstring,ending);
}

void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names, double downscale_factor) {
	if (dir_name == NULL) {
		return;
	}

	string dir_name_ = string(dir_name);
	vector<string> files_;

	DIR *dp;
	struct dirent *ep;
	dp = opendir (dir_name);

	if (dp != NULL)
	{
		while (ep = readdir (dp)) {
			if (ep->d_name[0] != '.')
				files_.push_back(ep->d_name);
		}

		(void) closedir (dp);
	}
	else {
		cerr << ("Couldn't open the directory");
		return;
	}

	for (unsigned int i=0; i<files_.size(); i++) {
		if (files_[i][0] == '.' || !(hasEndingLower(files_[i],"jpg")
														|| hasEndingLower(files_[i],"png")
														|| hasEndingLower(files_[i],"ppm")
														|| hasEndingLower(files_[i],"pgm"))) {
			continue;
		}
		cv::Mat m_ = cv::imread(string(dir_name_).append("/").append(files_[i]));
		if(downscale_factor != 1.0)
			cv::resize(m_,m_,Size(),downscale_factor,downscale_factor);
		images_names.push_back(files_[i]);
		images.push_back(m_);
	}
}
