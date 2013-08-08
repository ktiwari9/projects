#include "Detector.hpp"

DetectorBlackBox::DetectorBlackBox(detectorType type)
{
    switch(type)
    {
        case SIFT_ENUM:
        {
            double thresh = 0.04;
            double edge_thresh = 10;
            _myDetector = new SiftFeatureDetector(thresh, edge_thresh);
            break;
        }

        case SURF_ENUM:
        {
            int minHessian = 400;
            _myDetector = new SurfFeatureDetector( minHessian );
            break;
        }

        case STAR_ENUM:
            _myDetector = new StarFeatureDetector();
            break;

        case MSER_ENUM:
            _myDetector = new MserFeatureDetector();
            break;

        case FAST_ENUM:
            _myDetector = new FastFeatureDetector();
            break;

        default:
            cout << "Unknown detector type" << endl;
    }
}

int DetectorBlackBox::readImage(char* img)
{
    _myImg = imread(img, CV_LOAD_IMAGE_GRAYSCALE);

    if(!_myImg.data)
    {
        cout<< "Error: readImage: no image." << endl;
        return -1;
    }

    return 0;
}

int DetectorBlackBox::extractKeyPoints()
{
    _myDetector->detect( _myImg, _myKeypoints );

    return 0;
}

void DetectorBlackBox::showKeyPoints()
{
    Mat img_keypoints;
    drawKeypoints( _myImg, _myKeypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("Keypoints", img_keypoints );

    waitKey(0);
}

vector<KeyPoint> DetectorBlackBox::getKeyPoints()
{
    return _myKeypoints;
}
