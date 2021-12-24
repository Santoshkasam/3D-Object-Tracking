
#include <numeric>
#include "matching2D.hpp"

using namespace std;

/* Keypoint matching */
// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knnMatches;  
        // TODO : implement k-nearest-neighbor matching
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        // TODO : filter matches using descriptor distance ratio test
        for (auto it = knnMatches.begin(); it != knnMatches.end(); ++it)
        {
            float distThrushold = 0.8;
            if ((*it)[0].distance < distThrushold * (*it)[1].distance )
            {
                matches.push_back((*it)[0]);
            }
               
        }
        
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

/* Key point descriptors */

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) { 
        
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    } else if (descriptorType.compare("ORB") == 0) {
        
        extractor = cv::ORB::create();

    } else if (descriptorType.compare("FREAK") == 0) {
        
        extractor = cv::xfeatures2d::FREAK::create();

    } else if (descriptorType.compare("AKAZE") == 0) {

        extractor = cv::AKAZE::create();

    } else if (descriptorType.compare("SIFT") == 0) {

        extractor = cv::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

/* Keypoint detectors */

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the Harris Corner detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    /*
        The following function detects corners in the given image (img). It generates a 
        cv::KeyPoint for every corner that exceeds the min intensity thrushold. 
        
        Then keypoints which are the maxima of their local neighbourhood are recognized using 
        Non Maxima Suppression (NMS) and added to the keypoints vector.

        At the end, if bVis is true. The original image is cloned and the detected keypoints 
        are visualized.
    */
    
    // Detector parameters
    int blockSize = 4;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.

    cv::Mat nms_result = cv::Mat::zeros(img.size(), CV_32FC1);
    double maxOverlap = 0.0; // max permissible overlap between two features in %, used during NMS

    // non maxima suppression
    for (auto j = 0; j < dst_norm.rows; j++)
    {
        for (auto i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j,i);
            
            // consider a pixel only if its response value is greater than the threshold
            if( response > minResponse )
            {

                // store the current pixel as the keypoint. define the coordinates, size (neighbourhood diameter)
                // and the harris response value.
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i,j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppresion (NMS) in local neighbourhood around new key point.
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {
                            // if overlap is greater than zero and if the response of new kpt is higher
                            // than the old keypoint, then the new key point will replace the old 
                            // keypoint in the keypoints vector.
                            *it = newKeyPoint;
                            break;
                        }

                    }
                                                 
                    
                }

                // if the new keypoint does not overlap with any of the pre-existing keypoints
                // then add the new keypoint to the keypoints vector.
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }

            }           
            
        }
        
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris-corner detection  with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector; // This Feature Detector pointer is a place holder for a keypoint-detector
                                           // that is later assigned depending on the detector type

    // Following: all the detectors except FAST are created with default parameters
    
    if (detectorType.compare("FAST") == 0)
    {
       int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around it
       bool bNMS = true;
       cv::FastFeatureDetector::DetectorType FAST_type = cv::FastFeatureDetector::TYPE_9_16;
       detector = cv::FastFeatureDetector::create(threshold, bNMS, FAST_type); 
        
    } else if (detectorType.compare("BRISK") == 0) {

       detector = cv::BRISK::create();
        
    } else if (detectorType.compare("ORB") == 0) {
        
        detector = cv::ORB::create();
       
    } else if (detectorType.compare("AKAZE") == 0) {
       
        detector = cv::AKAZE::create();

    } else if (detectorType.compare("SIFT") == 0) {
        
        detector = cv::SIFT::create();
       
    } else {
        
        throw std::invalid_argument("No such detector is available");
    }
    
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    

    if(bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);    
    }
    
    
}