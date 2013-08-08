#include "FivePointsWrapper.h"

using namespace std;
using namespace cv;

FivePoints::FivePoints()
{
    srand ( time(NULL) );

    _min_error = 100;
    _num_iter = 2000;
}
/*
void FivePoints::compute(Mat calib_hpts1, Mat calib_hpts2, Mat_<double> &Ra, Mat_<double> &Rb, Mat_<double> &t)
{
    Matx33d E;
    cv::Mat_<double> error_matrix(calib_hpts2.cols,calib_hpts2.cols);
    double error;

    for(int iter=0; iter < _num_iter; ++iter)
    {
        // Compute best candidates for this round
        E = libmvFivePoints(calib_hpts1, calib_hpts2);

        error_matrix = calib_hpts1.t()*Mat(E)*calib_hpts2;
        error = 0;
        for(int i=0; i<calib_hpts2.cols; ++i)
            error += error_matrix(i,i);
        if( fabs(error) < _min_error )
        {
            _min_error = fabs(error);
            _best_E = Mat(E);
        }
    }

    essentialToRotationTranslation(_best_E, Ra, Rb, t);

}
*/
void FivePoints::compute(Mat calib_hpts1, Mat calib_hpts2, Mat_<double> &Ra, Mat_<double> &Rb, Mat_<double> &t)
{
    vector<Matx33d> E;
    cv::Mat_<double> error_matrix(calib_hpts1.cols,calib_hpts2.cols);
    int nbInliers = 0;
    int max_in = 0;
    int* inliers = (int*) malloc(calib_hpts1.cols*sizeof(int));
    double thresh = 1.0;
    int num_iter = 1000;

    for(int iter=0; iter < _num_iter; ++iter)
    {
        // Compute best candidates for this round
        E = libmvFivePoints(calib_hpts1, calib_hpts2);

        // Compute best candidates for this round
        for(int n=0; n<E.size(); ++n)
        {
            cout << "kikou1" << endl;
            error_matrix = calib_hpts1.t()*Mat(E[n])*calib_hpts2;
            cout << "kikou2" << endl;
            for(int i=0; i<error_matrix.cols; ++i)
            {
                 if(fabs(error_matrix(i,i)) < thresh)
                 {
                     inliers[nbInliers] = i;
                     ++nbInliers;
                 }
            }
            if(nbInliers > max_in)
            {
                _best_E = E[n];
            }
        }
    }

    essentialToRotationTranslation(_best_E, Ra, Rb, t);
}

//Matx33d FivePoints::libmvFivePoints(Mat_<double> calib_hpts1, Mat_<double> calib_hpts2)
vector<Matx33d> FivePoints::libmvFivePoints(Mat_<double> calib_hpts1, Mat_<double> calib_hpts2)
{
    libmv::vector<libmv::Mat3> E;
    std::vector< Matx33d > cvE;
    libmv::Mat2X Q1(2,5);
    libmv::Mat2X Q2(2,5);
    Mat_<double> HQ1(3, 5);
    Mat_<double> HQ2(3, 5);
    int l;

    // de homogeneise calib_hpts and compy the
    // randomly selected ones to a matrix.
    for(int n=0; n<5; ++n)
    {
        l = rand() % calib_hpts1.cols;
        Q1(0, n) = calib_hpts1.at<double>(0, l);
        Q1(1, n) = calib_hpts1.at<double>(1, l);
        Q2(0, n) = calib_hpts2.at<double>(0, l);
        Q2(1, n) = calib_hpts2.at<double>(1, l);

        HQ1(0, n) = calib_hpts1.at<double>(0, l);
        HQ1(1, n) = calib_hpts1.at<double>(0, l);
        HQ1(2, n) = 1;
        HQ2(0, n) = calib_hpts2.at<double>(0, l);
        HQ2(1, n) = calib_hpts2.at<double>(0, l);
        HQ2(2, n) = 1;
    }

    // Actually compute E
    libmv::FivePointsRelativePose(Q1, Q2, &E);

    // transform it to an opencv matrix
    cvE.resize(E.size());
    for(int k=0; k<E.size(); ++k)
    {
        for(int i=0; i<3; ++i)
            for(int j=0; j<3; ++j)
                cvE[k](i,j) = E[k](i,j);
    }

    //return getBestCandidate(cvE, HQ1, HQ2);
    return cvE;
}

Matx33d FivePoints::getBestCandidate(std::vector< Matx33d > candidates, Mat_<double> HQ1, Mat_<double> HQ2)
{
    Matx33d best_candidate;
    Mat_<double> candidates_error_matrix(HQ1.cols, HQ1.cols);
    double min_candidates_error = 1000;
    double candidate_error;

    // choose best E based on reprojection error
    // e = Q1'*E*Q2
    for(int k=0; k<candidates.size(); ++k)
    {
        candidates_error_matrix = HQ1.t()*Mat_<double>(candidates[k])*HQ2;

        candidate_error = 0;
        for(int n=0; n<HQ1.cols; ++n)
        candidate_error += candidates_error_matrix(n,n);

        if( fabs(candidate_error) < min_candidates_error )
        {
            min_candidates_error = fabs(candidate_error);
            best_candidate = candidates[k];
        }
    }

    return best_candidate;
}

//void FivePoints::essentialToRotationTranslation(Mat_<double> E, Mat_<double> &Ra, Mat_<double> &Rb, Mat_<double> &t)
void FivePoints::essentialToRotationTranslation(Matx33d E, Mat_<double> &Ra, Mat_<double> &Rb, Mat_<double> &t)
{
    SVD svd(E);
    Matx33d W(0,-1,0,   //HZ 9.13
            1,0,0,
            0,0,1);
    Matx33d Winv(0,1,0,
                -1,0,0,
                0,0,1);
    Ra = svd.u * Mat(W) * svd.vt; //HZ 9.19
    Rb = svd.u * Mat(Winv) * svd.vt; //HZ 9.19
    t = svd.u.col(2); //u3
}

double FivePoints::getMinError()
{
    return _min_error;
}

/*
start =cvGetTickCount();
end = cvGetTickCount();
cout << "Time :" << (end - start)/(1e3*cvGetTickFrequency())<< endl;

cout << "det U = " << determinant(svd.u) << endl;
cout << "det V = " << determinant(svd.vt) << endl;
cout << determinant(cvE) << endl;

double tr=trace(cvE*cvE.t())[0];
Mat e = 2.0*cvE*cvE.t()*cvE - tr*cvE;
cout << "Trace" << endl;
print(e);
*/
