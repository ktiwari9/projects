#include "FindCameraMatrices3D2D.h"

using namespace std;
using namespace cv;

bool FindCameraMatrices3D2D(cv::Mat_<double>& rvec,
                              cv::Mat_<double>& t,
                              cv::Mat_<double>& R,
                              std::vector<cv::Point3f> ppcloud,
                              std::vector<cv::Point2f> imgPoints,
                              bool use_gpu,
                              Mat K,
                              Mat distortion_coeff)
{
	if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
		//something went wrong aligning 3D to 2D points..
		cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
		return false;
	}

	vector<int> inliers;
	if(!use_gpu) {
		//use CPU
		double minVal,maxVal; cv::minMaxIdx(imgPoints,&minVal,&maxVal);
		CV_PROFILE("solvePnPRansac",cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, 1000, 0.01 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, cv::EPNP);)
		//CV_PROFILE("solvePnP",cv::solvePnP(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, CV_EPNP);)
	}/* else {
		//use GPU ransac
		//make sure datatstructures are cv::gpu compatible
		cv::Mat ppcloud_m(ppcloud); ppcloud_m = ppcloud_m.t();
		cv::Mat imgPoints_m(imgPoints); imgPoints_m = imgPoints_m.t();
		cv::Mat rvec_,t_;

		cv::gpu::solvePnPRansac(ppcloud_m,imgPoints_m,K_32f,distcoeff_32f,rvec_,t_,false);

		rvec_.convertTo(rvec,CV_64FC1);
		t_.convertTo(t,CV_64FC1);
	}*/

	vector<cv::Point2f> projected3D;
	cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);

	if(inliers.size()==0) { //get inliers
		for(int i=0;i<projected3D.size();i++) {
			if(norm(projected3D[i]-imgPoints[i]) < 10.0)
				inliers.push_back(i);
		}
	}

	if(inliers.size() < (double)(imgPoints.size())/5.0) {
		cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
		return false;
	}

	if(cv::norm(t) > 200.0) {
		// this is bad...
		cerr << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}

	cv::Rodrigues(rvec, R);
	if(!CheckCoherentRotation(R)) {
		cerr << "rotation is incoherent. we should try a different base view..." << endl;
		return false;
	}

	std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
	return true;
}
