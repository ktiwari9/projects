#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"

#include <fstream>
#include <iostream>
#include <memory>

#if defined WIN32 || defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#include "sys/types.h"
#endif
#include <sys/stat.h>

#define DEBUG_DESC_PROGRESS

using namespace cv;
using namespace std;

const string paramsFile = "params.xml";
const string vocabularyFile = "vocabulary.xml.gz";
const string bowImageDescriptorsDir = "/bowImageDescriptors";
const string svmsDir = "/svms";
const string plotsDir = "/plots";

void makeDir( const string& dir )
{
#if defined WIN32 || defined _WIN32
    CreateDirectory( dir.c_str(), 0 );
#else
    mkdir( dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
#endif
}

void makeUsedDirs( const string& rootPath )
{
    makeDir(rootPath + bowImageDescriptorsDir);
    makeDir(rootPath + svmsDir);
    makeDir(rootPath + plotsDir);
}

/****************************************************************************************\
*                    Classes to work with PASCAL VOC dataset                             *
\****************************************************************************************/
//
// TODO: refactor this part of the code
//


//used to specify the (sub-)dataset over which operations are performed
enum ObdDatasetType {CV_OBD_TRAIN, CV_OBD_TEST};

class ObdObject
{
public:
    string object_class;
    Rect boundingBox;
};

//extended object data specific to VOC
enum VocPose {CV_VOC_POSE_UNSPECIFIED, CV_VOC_POSE_FRONTAL, CV_VOC_POSE_REAR, CV_VOC_POSE_LEFT, CV_VOC_POSE_RIGHT};
class VocObjectData
{
public:
    bool difficult;
    bool occluded;
    bool truncated;
    VocPose pose;
};
//enum VocDataset {CV_VOC2007, CV_VOC2008, CV_VOC2009, CV_VOC2010};
enum VocPlotType {CV_VOC_PLOT_SCREEN, CV_VOC_PLOT_PNG};
enum VocGT {CV_VOC_GT_NONE, CV_VOC_GT_DIFFICULT, CV_VOC_GT_PRESENT};
enum VocConfCond {CV_VOC_CCOND_RECALL, CV_VOC_CCOND_SCORETHRESH};
enum VocTask {CV_VOC_TASK_CLASSIFICATION, CV_VOC_TASK_DETECTION};

class ObdImage
{
public:
    ObdImage(string p_id, string p_path) : id(p_id), path(p_path) {}
    string id;
    string path;
};

//used by getDetectorGroundTruth to sort a two dimensional list of floats in descending order
class ObdScoreIndexSorter
{
public:
    float score;
    int image_idx;
    int obj_idx;
    bool operator < (const ObdScoreIndexSorter& compare) const {return (score < compare.score);}
};

/****************************************************************************************\
*                            Sample on image classification                             *
\****************************************************************************************/
//
// This part of the code was a little refactor
//
struct DDMParams
{
    DDMParams() : detectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce") {}
    DDMParams( const string _detectorType, const string _descriptorType, const string& _matcherType ) :
        detectorType(_detectorType), descriptorType(_descriptorType), matcherType(_matcherType){}
    void read( const FileNode& fn )
    {
        fn["detectorType"] >> detectorType;
        fn["descriptorType"] >> descriptorType;
        fn["matcherType"] >> matcherType;
    }
    void write( FileStorage& fs ) const
    {
        fs << "detectorType" << detectorType;
        fs << "descriptorType" << descriptorType;
        fs << "matcherType" << matcherType;
    }
    void print() const
    {
        cout << "detectorType: " << detectorType << endl;
        cout << "descriptorType: " << descriptorType << endl;
        cout << "matcherType: " << matcherType << endl;
    }

    string detectorType;
    string descriptorType;
    string matcherType;
};

struct VocabTrainParams
{
    VocabTrainParams() : trainObjClass("chair"), vocabSize(1000), memoryUse(200), descProportion(0.3f) {}
    VocabTrainParams( const string _trainObjClass, size_t _vocabSize, size_t _memoryUse, float _descProportion ) :
            trainObjClass(_trainObjClass), vocabSize(_vocabSize), memoryUse(_memoryUse), descProportion(_descProportion) {}
    void read( const FileNode& fn )
    {
        fn["trainObjClass"] >> trainObjClass;
        fn["vocabSize"] >> vocabSize;
        fn["memoryUse"] >> memoryUse;
        fn["descProportion"] >> descProportion;
    }
    void write( FileStorage& fs ) const
    {
        fs << "trainObjClass" << trainObjClass;
        fs << "vocabSize" << vocabSize;
        fs << "memoryUse" << memoryUse;
        fs << "descProportion" << descProportion;
    }
    void print() const
    {
        cout << "trainObjClass: " << trainObjClass << endl;
        cout << "vocabSize: " << vocabSize << endl;
        cout << "memoryUse: " << memoryUse << endl;
        cout << "descProportion: " << descProportion << endl;
    }


    string trainObjClass; // Object class used for training visual vocabulary.
                          // It shouldn't matter which object class is specified here - visual vocab will still be the same.
    int vocabSize; //number of visual words in vocabulary to train
    int memoryUse; // Memory to preallocate (in MB) when training vocab.
                      // Change this depending on the size of the dataset/available memory.
    float descProportion; // Specifies the number of descriptors to use from each image as a proportion of the total num descs.
};

struct SVMTrainParamsExt
{
    SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {}
    SVMTrainParamsExt( float _descPercent, float _targetRatio, bool _balanceClasses ) :
            descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
    void read( const FileNode& fn )
    {
        fn["descPercent"] >> descPercent;
        fn["targetRatio"] >> targetRatio;
        fn["balanceClasses"] >> balanceClasses;
    }
    void write( FileStorage& fs ) const
    {
        fs << "descPercent" << descPercent;
        fs << "targetRatio" << targetRatio;
        fs << "balanceClasses" << balanceClasses;
    }
    void print() const
    {
        cout << "descPercent: " << descPercent << endl;
        cout << "targetRatio: " << targetRatio << endl;
        cout << "balanceClasses: " << balanceClasses << endl;
    }

    float descPercent; // Percentage of extracted descriptors to use for training.
    float targetRatio; // Try to get this ratio of positive to negative samples (minimum).
    bool balanceClasses;    // Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored).
};

Mat trainVocabulary( const string& filename, VocData& vocData, const VocabTrainParams& trainParams,
                     const Ptr<FeatureDetector>& fdetector, const Ptr<DescriptorExtractor>& dextractor )
{
    Mat vocabulary;
    if( !readVocabulary( filename, vocabulary) )
    {
        CV_Assert( dextractor->descriptorType() == CV_32FC1 );
        const int descByteSize = dextractor->descriptorSize()*4;
        const int maxDescCount = (trainParams.memoryUse * 1048576) / descByteSize; // Total number of descs to use for training.

        cout << "Extracting VOC data..." << endl;
        vector<ObdImage> images;
        vector<char> objectPresent;
        vocData.getClassImages( trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent );

        cout << "Computing descriptors..." << endl;
        RNG& rng = theRNG();
        TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowTrainer( trainParams.vocabSize, terminate_criterion, 3, KMEANS_PP_CENTERS );

        while( images.size() > 0 )
        {
            if( bowTrainer.descripotorsCount() >= maxDescCount )
            {
                assert( bowTrainer.descripotorsCount() == maxDescCount );
#ifdef DEBUG_DESC_PROGRESS
                cout << "Breaking due to full memory ( descriptors count = " << bowTrainer.descripotorsCount()
                        << "; descriptor size in bytes = " << descByteSize << "; all used memory = "
                        << bowTrainer.descripotorsCount()*descByteSize << endl;
#endif
                break;
            }

            // Randomly pick an image from the dataset which hasn't yet been seen
            // and compute the descriptors from that image.
            int randImgIdx = rng( images.size() );
            Mat colorImage = imread( images[randImgIdx].path );
            vector<KeyPoint> imageKeypoints;
            fdetector->detect( colorImage, imageKeypoints );
            Mat imageDescriptors;
            dextractor->compute( colorImage, imageKeypoints, imageDescriptors );

            //check that there were descriptors calculated for the current image
            if( !imageDescriptors.empty() )
            {
                int descCount = imageDescriptors.rows;
                // Extract trainParams.descProportion descriptors from the image, breaking if the 'allDescriptors' matrix becomes full
                int descsToExtract = static_cast<int>(trainParams.descProportion * static_cast<float>(descCount));
                // Fill mask of used descriptors
                vector<char> usedMask( descCount, false );
                fill( usedMask.begin(), usedMask.begin() + descsToExtract, true );
                for( int i = 0; i < descCount; i++ )
                {
                    int i1 = rng(descCount), i2 = rng(descCount);
                    char tmp = usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
                }

                for( int i = 0; i < descCount; i++ )
                {
                    if( usedMask[i] && bowTrainer.descripotorsCount() < maxDescCount )
                        bowTrainer.add( imageDescriptors.row(i) );
                }
            }

#ifdef DEBUG_DESC_PROGRESS
            cout << images.size() << " images left, " << images[randImgIdx].id << " processed - "
                    <</* descs_extracted << "/" << image_descriptors.rows << " extracted - " << */
                    cvRound((static_cast<double>(bowTrainer.descripotorsCount())/static_cast<double>(maxDescCount))*100.0)
                    << " % memory used" << ( imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "") << endl;
#endif

            // Delete the current element from images so it is not added again
            images.erase( images.begin() + randImgIdx );
        }

        cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: " << bowTrainer.descripotorsCount() << endl;

        cout << "Training vocabulary..." << endl;
        vocabulary = bowTrainer.cluster();

        if( !writeVocabulary(filename, vocabulary) )
        {
            cout << "Error: file " << filename << " can not be opened to write" << endl;
            exit(-1);
        }
    }
    return vocabulary;
}

// Load in the bag of words vectors for a set of images, from file if possible
void calculateImageDescriptors( const vector<ObdImage>& images, vector<Mat>& imageDescriptors,
                                Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
                                const string& resPath )
{
    CV_Assert( !bowExtractor->getVocabulary().empty() );
    imageDescriptors.resize( images.size() );

    for( size_t i = 0; i < images.size(); i++ )
    {
        string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";
        if( readBowImageDescriptor( filename, imageDescriptors[i] ) )
        {
#ifdef DEBUG_DESC_PROGRESS
            cout << "Loaded bag of word vector for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << endl;
#endif
        }
        else
        {
            Mat colorImage = imread( images[i].path );
#ifdef DEBUG_DESC_PROGRESS
            cout << "Computing descriptors for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << flush;
#endif
            vector<KeyPoint> keypoints;
            fdetector->detect( colorImage, keypoints );
#ifdef DEBUG_DESC_PROGRESS
                cout << " + generating BoW vector" << std::flush;
#endif
            bowExtractor->compute( colorImage, keypoints, imageDescriptors[i] );
#ifdef DEBUG_DESC_PROGRESS
            cout << " ...DONE " << static_cast<int>(static_cast<float>(i+1)/static_cast<float>(images.size())*100.0)
                 << " % complete" << endl;
#endif
            if( !imageDescriptors[i].empty() )
            {
                if( !writeBowImageDescriptor( filename, imageDescriptors[i] ) )
                {
                    cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
                    exit(-1);
                }
            }
        }
    }
}

void removeEmptyBowImageDescriptors( vector<ObdImage>& images, vector<Mat>& bowImageDescriptors,
                                     vector<char>& objectPresent )
{
    CV_Assert( !images.empty() );
    for( int i = (int)images.size() - 1; i >= 0; i-- )
    {
        bool res = bowImageDescriptors[i].empty();
        if( res )
        {
            cout << "Removing image " << images[i].id << " due to no descriptors..." << endl;
            images.erase( images.begin() + i );
            bowImageDescriptors.erase( bowImageDescriptors.begin() + i );
            objectPresent.erase( objectPresent.begin() + i );
        }
    }
}

void removeBowImageDescriptorsByCount( vector<ObdImage>& images, vector<Mat> bowImageDescriptors, vector<char> objectPresent,
                                       const SVMTrainParamsExt& svmParamsExt, int descsToDelete )
{
    RNG& rng = theRNG();
    int pos_ex = std::count( objectPresent.begin(), objectPresent.end(), (char)1 );
    int neg_ex = std::count( objectPresent.begin(), objectPresent.end(), (char)0 );

    while( descsToDelete != 0 )
    {
        int randIdx = rng(images.size());

        // Prefer positive training examples according to svmParamsExt.targetRatio if required
        if( objectPresent[randIdx] )
        {
            if( (static_cast<float>(pos_ex)/static_cast<float>(neg_ex+pos_ex)  < svmParamsExt.targetRatio) &&
                (neg_ex > 0) && (svmParamsExt.balanceClasses == false) )
            { continue; }
            else
            { pos_ex--; }
        }
        else
        { neg_ex--; }

        images.erase( images.begin() + randIdx );
        bowImageDescriptors.erase( bowImageDescriptors.begin() + randIdx );
        objectPresent.erase( objectPresent.begin() + randIdx );

        descsToDelete--;
    }
    CV_Assert( bowImageDescriptors.size() == objectPresent.size() );
}

void setSVMParams( CvSVMParams& svmParams, CvMat& class_wts_cv, const Mat& responses, bool balanceClasses )
{
    int pos_ex = countNonZero(responses == 1);
    int neg_ex = countNonZero(responses == -1);
    cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples" << endl;

    svmParams.svm_type = CvSVM::C_SVC;
    svmParams.kernel_type = CvSVM::RBF;
    if( balanceClasses )
    {
        Mat class_wts( 2, 1, CV_32FC1 );
        // The first training sample determines the '+1' class internally, even if it is negative,
        // so store whether this is the case so that the class weights can be reversed accordingly.
        bool reversed_classes = (responses.at<float>(0) < 0.f);
        if( reversed_classes == false )
        {
            class_wts.at<float>(0) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
            class_wts.at<float>(1) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
        }
        else
        {
            class_wts.at<float>(0) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex);
            class_wts.at<float>(1) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex);
        }
        class_wts_cv = class_wts;
        svmParams.class_weights = &class_wts_cv;
    }
}

void setSVMTrainAutoParams( CvParamGrid& c_grid, CvParamGrid& gamma_grid,
                            CvParamGrid& p_grid, CvParamGrid& nu_grid,
                            CvParamGrid& coef_grid, CvParamGrid& degree_grid )
{
    c_grid = CvSVM::get_default_grid(CvSVM::C);

    gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

    p_grid = CvSVM::get_default_grid(CvSVM::P);
    p_grid.step = 0;

    nu_grid = CvSVM::get_default_grid(CvSVM::NU);
    nu_grid.step = 0;

    coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
    coef_grid.step = 0;

    degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
    degree_grid.step = 0;
}

void trainSVMClassifier( CvSVM& svm, const SVMTrainParamsExt& svmParamsExt, const string& objClassName, VocData& vocData,
                         Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
                         const string& resPath )
{
    /* first check if a previously trained svm for the current class has been saved to file */
    string svmFilename = resPath + svmsDir + "/" + objClassName + ".xml.gz";

    FileStorage fs( svmFilename, FileStorage::READ);
    if( fs.isOpened() )
    {
        cout << "*** LOADING SVM CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        svm.load( svmFilename.c_str() );
    }
    else
    {
        cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "..." << endl;

        // Get classification ground truth for images in the training set
        vector<ObdImage> images;
        vector<Mat> bowImageDescriptors;
        vector<char> objectPresent;
        vocData.getClassImages( objClassName, CV_OBD_TRAIN, images, objectPresent );

        // Compute the bag of words vector for each image in the training set.
        calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );

        // Remove any images for which descriptors could not be calculated
        removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent );

        CV_Assert( svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f );
        if( svmParamsExt.descPercent < 1.f )
        {
            int descsToDelete = static_cast<int>(static_cast<float>(images.size())*(1.0-svmParamsExt.descPercent));

            cout << "Using " << (images.size() - descsToDelete) << " of " << images.size() <<
                    " descriptors for training (" << svmParamsExt.descPercent*100.0 << " %)" << endl;
            removeBowImageDescriptorsByCount( images, bowImageDescriptors, objectPresent, svmParamsExt, descsToDelete );
        }

        // Prepare the input matrices for SVM training.
        Mat trainData( images.size(), bowExtractor->getVocabulary().rows, CV_32FC1 );
        Mat responses( images.size(), 1, CV_32SC1 );

        // Transfer bag of words vectors and responses across to the training data matrices
        for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
        {
            // Transfer image descriptor (bag of words vector) to training data matrix
            Mat submat = trainData.row(imageIdx);
            if( bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize() )
            {
                cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
                     << " differs from vocabulary size" << bowExtractor->getVocabulary().cols << endl;
                exit(-1);
            }
            bowImageDescriptors[imageIdx].copyTo( submat );

            // Set response value
            responses.at<int>(imageIdx) = objectPresent[imageIdx] ? 1 : -1;
        }

        cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << endl;
        CvSVMParams svmParams;
        CvMat class_wts_cv;
        setSVMParams( svmParams, class_wts_cv, responses, svmParamsExt.balanceClasses );
        CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
        setSVMTrainAutoParams( c_grid, gamma_grid,  p_grid, nu_grid, coef_grid, degree_grid );
        svm.train_auto( trainData, responses, Mat(), Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );
        cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << endl;

        svm.save( svmFilename.c_str() );
        cout << "SAVED CLASSIFIER TO FILE" << endl;
    }
}


int main(int argc, char** argv)
{
    if( argc != 3 && argc != 6 )
    {
    	help(argv);
        return -1;
    }

    const string vocPath = argv[1], resPath = argv[2];

    // Read or set default parameters
    string vocName;
    DDMParams ddmParams;
    VocabTrainParams vocabTrainParams;
    SVMTrainParamsExt svmTrainParamsExt;

    makeUsedDirs( resPath );

    FileStorage paramsFS( resPath + "/" + paramsFile, FileStorage::READ );
    if( paramsFS.isOpened() )
    {
       readUsedParams( paramsFS.root(), vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
       CV_Assert( vocName == getVocName(vocPath) );
    }
    else
    {
        vocName = getVocName(vocPath);
        if( argc!= 6 )
        {
            cout << "Feature detector, descriptor extractor, descriptor matcher must be set" << endl;
            return -1;
        }
        ddmParams = DDMParams( argv[3], argv[4], argv[5] ); // from command line
        // vocabTrainParams and svmTrainParamsExt is set by defaults
        paramsFS.open( resPath + "/" + paramsFile, FileStorage::WRITE );
        if( paramsFS.isOpened() )
        {
            writeUsedParams( paramsFS, vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
            paramsFS.release();
        }
        else
        {
            cout << "File " << (resPath + "/" + paramsFile) << "can not be opened to write" << endl;
            return -1;
        }
    }

    // Create detector, descriptor, matcher.
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create( ddmParams.detectorType );
    Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( ddmParams.descriptorType );
    Ptr<BOWImgDescriptorExtractor> bowExtractor;
    if( featureDetector.empty() || descExtractor.empty() )
    {
        cout << "featureDetector or descExtractor was not created" << endl;
        return -1;
    }
    {
        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( ddmParams.matcherType );
        if( featureDetector.empty() || descExtractor.empty() || descMatcher.empty() )
        {
            cout << "descMatcher was not created" << endl;
            return -1;
        }
        bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );
    }

    // Print configuration to screen
    printUsedParams( vocPath, resPath, ddmParams, vocabTrainParams, svmTrainParamsExt );
    // Create object to work with VOC
    VocData vocData( vocPath, false );

    // 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
    Mat vocabulary = trainVocabulary( resPath + "/" + vocabularyFile, vocData, vocabTrainParams,
                                      featureDetector, descExtractor );
    bowExtractor->setVocabulary( vocabulary );

    // 2. Train a classifier and run a sample query for each object class
    const vector<string>& objClasses = vocData.getObjectClasses(); // object class list
    for( size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx )
    {
        // Train a classifier on train dataset
        CvSVM svm;
        trainSVMClassifier( svm, svmTrainParamsExt, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );

        // Now use the classifier over all images on the test dataset and rank according to score order
        // also calculating precision-recall etc.
        computeConfidences( svm, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );
        // Calculate precision/recall/ap and use GNUPlot to output to a pdf file
        computeGnuPlotOutput( resPath, objClasses[classIdx], vocData );
    }
    return 0;
}
